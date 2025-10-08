import os
import asyncio
import base64
import json
import uuid
import warnings
import sounddevice as sd
import pytz
import random
import hashlib
import datetime
import time
import inspect
import numpy as np
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

# Suppress warnings
warnings.filterwarnings("ignore")

# Audio configuration
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 1024  # Number of frames per buffer
SAMPLE_DTYPE = 'int16'

# Debug mode flag
DEBUG = False

def debug_print(message):
    """Print only if debug mode is enabled"""
    if DEBUG:
        functionName = inspect.stack()[1].function
        if  functionName == 'time_it' or functionName == 'time_it_async':
            functionName = inspect.stack()[2].function
        print('{:%Y-%m-%d %H:%M:%S.%f}'.format(datetime.datetime.now())[:-3] + ' ' + functionName + ' ' + message)

def time_it(label, methodToRun):
    start_time = time.perf_counter()
    result = methodToRun()
    end_time = time.perf_counter()
    debug_print(f"Execution time for {label}: {end_time - start_time:.4f} seconds")
    return result

async def time_it_async(label, methodToRun):
    start_time = time.perf_counter()
    result = await methodToRun()
    end_time = time.perf_counter()
    debug_print(f"Execution time for {label}: {end_time - start_time:.4f} seconds")
    return result

class ToolProcessor:
    def __init__(self):
        # ThreadPoolExecutor could be used for complex implementations
        self.tasks = {}
    
    async def process_tool_async(self, tool_name, tool_content):
        """Process a tool call asynchronously and return the result"""
        # Create a unique task ID
        task_id = str(uuid.uuid4())
        
        # Create and store the task
        task = asyncio.create_task(self._run_tool(tool_name, tool_content))
        self.tasks[task_id] = task
        
        try:
            # Wait for the task to complete
            result = await task
            return result
        finally:
            # Clean up the task reference
            if task_id in self.tasks:
                del self.tasks[task_id]
    
    async def _run_tool(self, tool_name, tool_content):
        """Internal method to execute the tool logic"""
        debug_print(f"Processing tool: {tool_name}")
        tool = tool_name.lower()
        
        if tool == "getdateandtimetool":
            # Get current date in PST timezone
            pst_timezone = pytz.timezone("America/Los_Angeles")
            pst_date = datetime.datetime.now(pst_timezone)
            
            return {
                "formattedTime": pst_date.strftime("%I:%M %p"),
                "date": pst_date.strftime("%Y-%m-%d"),
                "year": pst_date.year,
                "month": pst_date.month,
                "day": pst_date.day,
                "dayOfWeek": pst_date.strftime("%A").upper(),
                "timezone": "PST"
            }
        
        elif tool == "trackordertool":
            # Simulate a long-running operation
            debug_print(f"TrackOrderTool starting operation that will take time...")
            await asyncio.sleep(10)  # Non-blocking sleep to simulate processing time
            
            # Extract order ID from toolUseContent
            content = tool_content.get("content", {})
            content_data = json.loads(content)
            order_id = content_data.get("orderId", "")
            request_notifications = content_data.get("requestNotifications", False)
            
            # Convert order_id to string if it's an integer
            if isinstance(order_id, int):
                order_id = str(order_id)
            # Validate order ID format
            if not order_id or not isinstance(order_id, str):
                return {
                    "error": "Invalid order ID format",
                    "orderStatus": "",
                    "estimatedDelivery": "",
                    "lastUpdate": ""
                }
            
            # Create deterministic randomness based on order ID
            # This ensures the same order ID always returns the same status
            seed = int(hashlib.md5(order_id.encode(), usedforsecurity=False).hexdigest(), 16) % 10000
            random.seed(seed)
            
            # Rest of the order tracking logic
            statuses = [
                "Order received", 
                "Processing", 
                "Preparing for shipment",
                "Shipped",
                "In transit", 
                "Out for delivery",
                "Delivered",
                "Delayed"
            ]
            
            weights = [10, 15, 15, 20, 20, 10, 5, 3]
            status = random.choices(statuses, weights=weights, k=1)[0]
            
            # Generate delivery date logic
            today = datetime.datetime.now()
            if status == "Delivered":
                delivery_days = -random.randint(0, 3)
                estimated_delivery = (today + datetime.timedelta(days=delivery_days)).strftime("%Y-%m-%d")
            elif status == "Out for delivery":
                estimated_delivery = today.strftime("%Y-%m-%d")
            else:
                delivery_days = random.randint(1, 10)
                estimated_delivery = (today + datetime.timedelta(days=delivery_days)).strftime("%Y-%m-%d")

            # Handle notification request
            notification_message = ""
            if request_notifications and status != "Delivered":
                notification_message = f"You will receive notifications for order {order_id}"

            # Return tracking information
            tracking_info = {
                "orderStatus": status,
                "orderNumber": order_id,
                "notificationStatus": notification_message
            }

            # Add appropriate fields based on status
            if status == "Delivered":
                tracking_info["deliveredOn"] = estimated_delivery
            elif status == "Out for delivery":
                tracking_info["expectedDelivery"] = "Today"
            else:
                tracking_info["estimatedDelivery"] = estimated_delivery

            # Add location information based on status
            if status == "In transit":
                tracking_info["currentLocation"] = "Distribution Center"
            elif status == "Delivered":
                tracking_info["deliveryLocation"] = "Front Door"
                
            # Add additional info for delayed status
            if status == "Delayed":
                tracking_info["additionalInfo"] = "Weather delays possible"
                
            debug_print(f"TrackOrderTool completed successfully")
            return tracking_info
        else:
            return {
                "error": f"Unsupported tool: {tool_name}"
            }

class BedrockStreamManager:
    """Manages bidirectional streaming with AWS Bedrock using asyncio"""
    
    # Event templates
    START_SESSION_EVENT = '''{
        "event": {
            "sessionStart": {
            "inferenceConfiguration": {
                "maxTokens": 1024,
                "topP": 0.9,
                "temperature": 0.7
                }
            }
        }
    }'''

    CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
            "promptName": "%s",
            "contentName": "%s",
            "type": "AUDIO",
            "interactive": true,
            "role": "USER",
            "audioInputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": 16000,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "audioType": "SPEECH",
                "encoding": "base64"
                }
            }
        }
    }'''

    AUDIO_EVENT_TEMPLATE = '''{
        "event": {
            "audioInput": {
            "promptName": "%s",
            "contentName": "%s",
            "content": "%s"
            }
        }
    }'''

    TEXT_CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
            "promptName": "%s",
            "contentName": "%s",
            "type": "TEXT",
            "role": "%s",
            "interactive": false,
                "textInputConfiguration": {
                    "mediaType": "text/plain"
                }
            }
        }
    }'''

    TEXT_INPUT_EVENT = '''{
        "event": {
            "textInput": {
            "promptName": "%s",
            "contentName": "%s",
            "content": "%s"
            }
        }
    }'''

    TOOL_CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
                "promptName": "%s",
                "contentName": "%s",
                "interactive": false,
                "type": "TOOL",
                "role": "TOOL",
                "toolResultInputConfiguration": {
                    "toolUseId": "%s",
                    "type": "TEXT",
                    "textInputConfiguration": {
                        "mediaType": "text/plain"
                    }
                }
            }
        }
    }'''

    CONTENT_END_EVENT = '''{
        "event": {
            "contentEnd": {
            "promptName": "%s",
            "contentName": "%s"
            }
        }
    }'''

    PROMPT_END_EVENT = '''{
        "event": {
            "promptEnd": {
            "promptName": "%s"
            }
        }
    }'''

    SESSION_END_EVENT = '''{
        "event": {
            "sessionEnd": {}
        }
    }'''
    
    def start_prompt(self):
        """Create a promptStart event"""
        get_default_tool_schema = json.dumps({
            "type": "object",
            "properties": {},
            "required": []
        })

        get_order_tracking_schema = json.dumps({
            "type": "object",
            "properties": {
                "orderId": {
                    "type": "string",
                    "description": "The order number or ID to track"
                },
                "requestNotifications": {
                    "type": "boolean",
                    "description": "Whether to set up notifications for this order",
                    "default": False
                }
            },
            "required": ["orderId"]
        })

        
        prompt_start_event = {
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {
                        "mediaType": "text/plain"
                    },
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "lupe",
                        "encoding": "base64",
                        "audioType": "SPEECH"
                    },
                    "toolUseOutputConfiguration": {
                        "mediaType": "application/json"
                    },
                    "toolConfiguration": {
                        "tools": [
                            {
                                "toolSpec": {
                                    "name": "getDateAndTimeTool",
                                    "description": "get information about the current date and time",
                                    "inputSchema": {
                                        "json": get_default_tool_schema
                                    }
                                }
                            },
                            {
                                "toolSpec": {
                                    "name": "trackOrderTool",
                                    "description": "Retrieves real-time order tracking information and detailed status updates for customer orders by order ID. Provides estimated delivery dates. Use this tool when customers ask about their order status or delivery timeline.",
                                    "inputSchema": {
                                    "json": get_order_tracking_schema
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        return json.dumps(prompt_start_event)
    
    def tool_result_event(self, content_name, content, role):
        """Create a tool result event"""

        if isinstance(content, dict):
            content_json_string = json.dumps(content)
        else:
            content_json_string = content
            
        tool_result_event = {
            "event": {
                "toolResult": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": content_json_string
                }
            }
        }
        return json.dumps(tool_result_event)
   
    def __init__(self, model_id='amazon.nova-sonic-v1:0', region='us-east-1'):
        """Initialize the stream manager."""
        self.model_id = model_id
        self.region = region
        
        # Replace RxPy subjects with asyncio queues
        self.audio_input_queue = asyncio.Queue()
        self.audio_output_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        
        self.response_task = None
        self.stream_response = None
        self.is_active = False
        self.barge_in = False
        self.bedrock_client = None
        
        # Audio playback components
        self.audio_player = None
        
        # Text response components
        self.display_assistant_text = False
        self.role = None

        # Session information
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.toolUseContent = ""
        self.toolUseId = ""
        self.toolName = ""

        # Add a tool processor
        self.tool_processor = ToolProcessor()
        
        # Add tracking for in-progress tool calls
        self.pending_tool_tasks = {}

        # Track whether audio content has actually started
        self.audio_stream_started = False

    def _initialize_client(self):
        """Initialize the Bedrock client."""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)
    
    async def initialize_stream(self):
        """Initialize the bidirectional stream with Bedrock."""
        if not self.bedrock_client:
            self._initialize_client()
        
        try:
            self.stream_response = await time_it_async("invoke_model_with_bidirectional_stream", lambda : self.bedrock_client.invoke_model_with_bidirectional_stream( InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)))
            self.is_active = True
            self.audio_stream_started = False
            default_system_prompt = """
            Eres un asistente conversacional profesional cuyo rol principal es asistir sobre seguros orientados a salud. Debes seguir estas reglas estrictas en TODAS las interacciones:

            ---

            1. IDIOMA Y TONO  
            - Habla siempre en español.  
            - Sé muy amable, empático y respetuoso. Usa expresiones como:  
            “Por supuesto”, “Con gusto”, “Gracias por compartirlo”, “¿En qué más puedo ayudarte?”.  
            - Evita jerga innecesaria y explica términos técnicos si el usuario lo pide.

            ---

            2. PRESENTACIÓN Y PERSONALIZACIÓN  
            - Al inicio de la conversación (o cuando sea natural), pregunta el nombre del usuario para personalizar la experiencia:  
            “¿Me puedes decir tu nombre para dirigirme a ti con más cercanía?”  
            - Si el usuario no quiere o da un nombre inválido, responde:  
            “No pasa nada si prefieres no dar tu nombre, seguiré ayudándote con mucho gusto.”

            ---

            3. ENFOQUE PRINCIPAL Y PROACTIVIDAD  
            - Tu foco es seguros de salud.  
            - Siempre que sea relevante, menciona tus capacidades:  
            “Puedo explicarte coberturas, exclusiones, pasos para un reclamo o ayudarte a entender tu póliza.”  
            - Sugiere temas relacionados:  
            “¿Quieres que revise coberturas, deducibles o cómo presentar un siniestro?”  
            - Si el usuario habla de otro tipo de seguro (auto, vida, etc.), puedes responder brevemente pero regresa al foco principal:  
            “También puedo orientarte en eso, aunque mi especialidad es salud. ¿Deseas que lo abordemos de forma general o regresamos al tema de salud?”

            ---

            4. SIMULACIONES O RESPUESTAS FICTICIAS (DEMO)  
            - Puedes ofrecer respuestas simuladas, pero deben estar claramente delimitadas y no deben presentarse como oficiales o válidas.  
            - Usa estos delimitadores:  
            --- INICIO SIMULACIÓN / DEMO ---  
            [contenido ficticio]  
            --- FIN SIMULACIÓN / DEMO ---  
            - Nunca inventes datos sensibles reales (números de póliza, CURP, direcciones o información personal verificable).  
            - Ejemplo:  
            --- INICIO SIMULACIÓN / DEMO ---  
            ID de póliza: 45345 (simulada)  
            Cobertura principal: Consultas médicas ambulatorias hasta $50,000 MXN anuales  
            Deducible: $2,000 por evento  
            Exclusiones comunes: procedimientos estéticos y enfermedades preexistentes no declaradas  
            --- FIN SIMULACIÓN / DEMO ---  
            Esta información es solo de demostración. Contacta a tu aseguradora para datos reales.

            ---

            5. GUARDRAILS DE SEGURIDAD Y ÉTICA  
            - No facilites ni apoyes fraude, falsificación ni acciones ilegales.  
            - No proporciones asesoría legal, médica o financiera vinculante.  
            Usa un aviso como:  
            “Esta es una orientación general, no sustituye el consejo de un profesional ni la información oficial de tu póliza.”  
            - No solicites ni almacenes datos sensibles innecesarios. Si el usuario los comparte, responde:  
            “Por seguridad, no es necesario que compartas esa información aquí. Puedo explicarte cómo enviarla de forma segura a tu aseguradora.”  
            - Si el usuario pide acciones reales (cancelar pólizas, presentar reclamos, acceder a sistemas), responde:  
            “No puedo ejecutar trámites reales ni acceder a sistemas de terceros, pero puedo guiarte paso a paso o redactarte el texto para que lo envíes.”

            ---

            6. MANEJO DE INCERTIDUMBRE Y VERIFICACIÓN  
            - Si no sabes algo, admítelo y ofrece opciones:  
            “No tengo esa información exacta. ¿Quieres que te muestre una simulación demo o cómo pedirla a tu aseguradora?”  
            - Cuando menciones coberturas, montos o plazos, aclara que deben verificarse con la aseguradora o contrato.

            ---

            7. ESTRUCTURA RECOMENDADA DE RESPUESTA  
            1. Saludo cordial.  
            2. Pregunta por nombre (si aún no se ha hecho).  
            3. Confirma el foco: “Puedo ayudarte con coberturas, deducibles o reclamos, ¿qué necesitas hoy?”  
            4. Respuesta clara, organizada y amable.  
            5. Si aplica, usa delimitadores de simulación (--- INICIO/FIN SIMULACIÓN / DEMO ---).  
            6. Cierre amable y proactivo: “¿Quieres que te prepare un ejemplo de correo o una simulación de reclamo?”

            ---

            8. PLANTILLAS ÚTILES  
            - Pregunta inicial:  
            “Antes de empezar, ¿cómo te puedo llamar?”  
            - Aviso de seguridad:  
            “Por seguridad, evita compartir contraseñas o números de identificación aquí.”  
            - Ejemplo de rechazo a solicitud no permitida:  
            “No puedo generar documentos oficiales ni comprobantes válidos. Solo la aseguradora puede hacerlo.”  
            - Ejemplo de ayuda alternativa:  
            “¿Deseas que te ayude a redactar un correo para tu aseguradora?”

            ---

            9. COMPORTAMIENTO ANTE ESCENARIOS FALSOS  
            - Mantén la conversación activa y coherente, incluso si el usuario inventa datos.  
            - Usa simulaciones controladas, pero recuerda los delimitadores y aclaraciones.  
            - Si el usuario pide algo imposible (emitir documentos oficiales, modificar pólizas), rechaza con respeto y explica por qué.

            ---

            10. PRIVACIDAD Y ALMACENAMIENTO  
            - No guardes ni recuerdes información personal fuera de la sesión sin permiso explícito.  
            - Si el usuario pide que “recuerdes” algo, aclara:  
            “Puedo recordarlo mientras estemos en esta sesión, pero no guardarlo de forma permanente.”

            ---

            11. RECORDATORIO FINAL  
            Tu misión es ofrecer una experiencia amable, clara, personalizada y ética, centrada en seguros de salud.  
            Si el usuario pide datos o acciones fuera del alcance, responde con orientación, no con ejecución.  
            Marca toda información ficticia con los delimitadores de simulación.

            ---

            """
            
            # Send initialization events
            prompt_event = self.start_prompt()
            text_content_start = self.TEXT_CONTENT_START_EVENT % (self.prompt_name, self.content_name, "SYSTEM")
            text_content = self.TEXT_INPUT_EVENT % (self.prompt_name, self.content_name, default_system_prompt)
            text_content_end = self.CONTENT_END_EVENT % (self.prompt_name, self.content_name)
            
            init_events = [self.START_SESSION_EVENT, prompt_event, text_content_start, text_content, text_content_end]
            
            for event in init_events:
                await self.send_raw_event(event)
                # Small delay between init events
                await asyncio.sleep(0.1)
            
            # Start listening for responses
            self.response_task = asyncio.create_task(self._process_responses())
            
            # Start processing audio input
            asyncio.create_task(self._process_audio_input())
            
            # Wait a bit to ensure everything is set up
            await asyncio.sleep(0.1)
            
            debug_print("Stream initialized successfully")
            return self
        except Exception as e:
            self.is_active = False
            print(f"Failed to initialize stream: {str(e)}")
            raise
    
    async def send_raw_event(self, event_json):
        """Send a raw event JSON to the Bedrock stream."""
        if not self.stream_response or not self.is_active:
            debug_print("Stream not initialized or closed")
            return
       
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        
        try:
            await self.stream_response.input_stream.send(event)
            # For debugging large events, you might want to log just the type
            if DEBUG:
                if len(event_json) > 200:
                    event_type = json.loads(event_json).get("event", {}).keys()
                    debug_print(f"Sent event type: {list(event_type)}")
                else:
                    debug_print(f"Sent event: {event_json}")
        except Exception as e:
            debug_print(f"Error sending event: {str(e)}")
            if DEBUG:
                import traceback
                traceback.print_exc()
    
    async def send_audio_content_start_event(self):
        """Send a content start event to the Bedrock stream."""
        content_start_event = self.CONTENT_START_EVENT % (self.prompt_name, self.audio_content_name)
        await self.send_raw_event(content_start_event)
    
    async def _process_audio_input(self):
        """Process audio input from the queue and send to Bedrock."""
        while self.is_active:
            try:
                # Get audio data from the queue
                data = await self.audio_input_queue.get()
                
                audio_bytes = data.get('audio_bytes')
                if not audio_bytes:
                    debug_print("No audio bytes received")
                    continue

                if not self.audio_stream_started:
                    await self.send_audio_content_start_event()
                    self.audio_stream_started = True
                
                # Base64 encode the audio data
                blob = base64.b64encode(audio_bytes)
                audio_event = self.AUDIO_EVENT_TEMPLATE % (
                    self.prompt_name, 
                    self.audio_content_name, 
                    blob.decode('utf-8')
                )
                
                # Send the event
                await self.send_raw_event(audio_event)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                debug_print(f"Error processing audio: {e}")
                if DEBUG:
                    import traceback
                    traceback.print_exc()
    
    def add_audio_chunk(self, audio_bytes):
        """Add an audio chunk to the queue."""
        self.audio_input_queue.put_nowait({
            'audio_bytes': audio_bytes,
            'prompt_name': self.prompt_name,
            'content_name': self.audio_content_name
        })
    
    async def send_audio_content_end_event(self):
        """Send a content end event to the Bedrock stream."""
        if not self.is_active or not self.audio_stream_started:
            debug_print("Stream is not active")
            return
        
        content_end_event = self.CONTENT_END_EVENT % (self.prompt_name, self.audio_content_name)
        await self.send_raw_event(content_end_event)
        debug_print("Audio ended")
        self.audio_stream_started = False
    
    async def send_tool_start_event(self, content_name, tool_use_id):
        """Send a tool content start event to the Bedrock stream."""
        content_start_event = self.TOOL_CONTENT_START_EVENT % (self.prompt_name, content_name, tool_use_id)
        debug_print(f"Sending tool start event: {content_start_event}")  
        await self.send_raw_event(content_start_event)

    async def send_tool_result_event(self, content_name, tool_result):
        """Send a tool content event to the Bedrock stream."""
        # Use the actual tool result from processToolUse
        tool_result_event = self.tool_result_event(content_name=content_name, content=tool_result, role="TOOL")
        debug_print(f"Sending tool result event: {tool_result_event}")
        await self.send_raw_event(tool_result_event)
    
    async def send_tool_content_end_event(self, content_name):
        """Send a tool content end event to the Bedrock stream."""
        tool_content_end_event = self.CONTENT_END_EVENT % (self.prompt_name, content_name)
        debug_print(f"Sending tool content event: {tool_content_end_event}")
        await self.send_raw_event(tool_content_end_event)
    
    async def send_prompt_end_event(self):
        """Close the stream and clean up resources."""
        if not self.is_active:
            debug_print("Stream is not active")
            return
        
        prompt_end_event = self.PROMPT_END_EVENT % (self.prompt_name)
        await self.send_raw_event(prompt_end_event)
        debug_print("Prompt ended")
        
    async def send_session_end_event(self):
        """Send a session end event to the Bedrock stream."""
        if not self.is_active:
            debug_print("Stream is not active")
            return

        await self.send_raw_event(self.SESSION_END_EVENT)
        self.is_active = False
        debug_print("Session ended")
    
    async def _process_responses(self):
        """Process incoming responses from Bedrock."""
        try:            
            while self.is_active:
                try:
                    output = await self.stream_response.await_output()
                    result = await output[1].receive()
                    if result.value and result.value.bytes_:
                        try:
                            response_data = result.value.bytes_.decode('utf-8')
                            json_data = json.loads(response_data)
                            
                            # Handle different response types
                            if 'event' in json_data:
                                if 'completionStart' in json_data['event']:
                                    debug_print(f"completionStart: {json_data['event']}")
                                elif 'contentStart' in json_data['event']:
                                    debug_print("Content start detected")
                                    content_start = json_data['event']['contentStart']
                                    # set role
                                    self.role = content_start['role']
                                    # Check for speculative content
                                    if 'additionalModelFields' in content_start:
                                        try:
                                            additional_fields = json.loads(content_start['additionalModelFields'])
                                            if additional_fields.get('generationStage') == 'SPECULATIVE':
                                                debug_print("Speculative content detected")
                                                self.display_assistant_text = True
                                            else:
                                                self.display_assistant_text = False
                                        except json.JSONDecodeError:
                                            debug_print("Error parsing additionalModelFields")
                                elif 'textOutput' in json_data['event']:
                                    text_content = json_data['event']['textOutput']['content']
                                    role = json_data['event']['textOutput']['role']
                                    # Check if there is a barge-in
                                    if '{ "interrupted" : true }' in text_content:
                                        debug_print("Barge-in detected. Stopping audio output.")
                                        self.barge_in = True

                                    if (self.role == "ASSISTANT" and self.display_assistant_text):
                                        print(f"Assistant: {text_content}")
                                    elif (self.role == "USER"):
                                        print(f"User: {text_content}")
                                elif 'audioOutput' in json_data['event']:
                                    audio_content = json_data['event']['audioOutput']['content']
                                    audio_bytes = base64.b64decode(audio_content)
                                    await self.audio_output_queue.put(audio_bytes)
                                elif 'toolUse' in json_data['event']:
                                    self.toolUseContent = json_data['event']['toolUse']
                                    self.toolName = json_data['event']['toolUse']['toolName']
                                    self.toolUseId = json_data['event']['toolUse']['toolUseId']
                                    debug_print(f"Tool use detected: {self.toolName}, ID: {self.toolUseId}")
                                elif 'contentEnd' in json_data['event'] and json_data['event'].get('contentEnd', {}).get('type') == 'TOOL':
                                    debug_print("Processing tool use and sending result")
                                     # Start asynchronous tool processing - non-blocking
                                    self.handle_tool_request(self.toolName, self.toolUseContent, self.toolUseId)
                                    debug_print("Processing tool use asynchronously")
                                elif 'contentEnd' in json_data['event']:
                                    debug_print("Content end")
                                elif 'completionEnd' in json_data['event']:
                                    # Handle end of conversation, no more response will be generated
                                    debug_print("End of response sequence")
                                elif 'usageEvent' in json_data['event']:
                                    debug_print(f"UsageEvent: {json_data['event']}")
                            # Put the response in the output queue for other components
                            await self.output_queue.put(json_data)
                        except json.JSONDecodeError:
                            await self.output_queue.put({"raw_data": response_data})
                except StopAsyncIteration:
                    # Stream has ended
                    break
                except Exception as e:
                   # Handle ValidationException properly
                    if "ValidationException" in str(e):
                        error_message = str(e)
                        print(f"Validation error: {error_message}")
                    else:
                        print(f"Error receiving response: {e}")
                    break
                    
        except Exception as e:
            print(f"Response processing error: {e}")
        finally:
            self.is_active = False

    def handle_tool_request(self, tool_name, tool_content, tool_use_id):
        """Handle a tool request asynchronously"""
        # Create a unique content name for this tool response
        tool_content_name = str(uuid.uuid4())
        
        # Create an asynchronous task for the tool execution
        task = asyncio.create_task(self._execute_tool_and_send_result(
            tool_name, tool_content, tool_use_id, tool_content_name))
        
        # Store the task
        self.pending_tool_tasks[tool_content_name] = task
        
        # Add error handling
        task.add_done_callback(
            lambda t: self._handle_tool_task_completion(t, tool_content_name))
    
    def _handle_tool_task_completion(self, task, content_name):
        """Handle the completion of a tool task"""
        # Remove task from pending tasks
        if content_name in self.pending_tool_tasks:
            del self.pending_tool_tasks[content_name]
        
        # Handle any exceptions
        if task.done() and not task.cancelled():
            exception = task.exception()
            if exception:
                debug_print(f"Tool task failed: {str(exception)}")
    
    async def _execute_tool_and_send_result(self, tool_name, tool_content, tool_use_id, content_name):
        """Execute a tool and send the result"""
        try:
            debug_print(f"Starting tool execution: {tool_name}")
            
            # Process the tool - this doesn't block the event loop
            tool_result = await self.tool_processor.process_tool_async(tool_name, tool_content)
            
            # Send the result sequence
            await self.send_tool_start_event(content_name, tool_use_id)
            await self.send_tool_result_event(content_name, tool_result)
            await self.send_tool_content_end_event(content_name)
            
            debug_print(f"Tool execution complete: {tool_name}")
        except Exception as e:
            debug_print(f"Error executing tool {tool_name}: {str(e)}")
            # Try to send an error response if possible
            try:
                error_result = {"error": f"Tool execution failed: {str(e)}"}
                
                await self.send_tool_start_event(content_name, tool_use_id)
                await self.send_tool_result_event(content_name, error_result)
                await self.send_tool_content_end_event(content_name)
            except Exception as send_error:
                debug_print(f"Failed to send error response: {str(send_error)}")
    
    async def close(self):
        """Close the stream properly."""
        if not self.is_active:
            return
        
        # Cancel any pending tool tasks
        for task in self.pending_tool_tasks.values():
            task.cancel()

        if self.response_task and not self.response_task.done():
            self.response_task.cancel()

        await self.send_audio_content_end_event()
        await self.send_prompt_end_event()
        await self.send_session_end_event()

        if self.stream_response:
            await self.stream_response.input_stream.close()

class AudioStreamer:
    """Handles continuous microphone input and audio output using separate streams."""
    
    def __init__(self, stream_manager):
        self.stream_manager = stream_manager
        self.is_streaming = False
        self.loop = asyncio.get_event_loop()
        
        # Initialize separate streams for input and output using sounddevice
        debug_print("Opening input audio stream...")
        self.input_stream = time_it(
            "AudioStreamerOpenInputStream",
            lambda: sd.InputStream(
                samplerate=INPUT_SAMPLE_RATE,
                channels=CHANNELS,
                dtype=SAMPLE_DTYPE,
                blocksize=CHUNK_SIZE,
                callback=self.input_callback
            )
        )
        debug_print("Input audio stream opened")

        debug_print("Opening output audio stream...")
        self.output_stream = time_it(
            "AudioStreamerOpenOutputStream",
            lambda: sd.OutputStream(
                samplerate=OUTPUT_SAMPLE_RATE,
                channels=CHANNELS,
                dtype=SAMPLE_DTYPE,
                blocksize=CHUNK_SIZE
            )
        )
        debug_print("Output audio stream opened")

    def input_callback(self, in_data, frames, time_info, status):
        """Callback function that schedules audio processing in the asyncio event loop"""
        if status:
            debug_print(f"Audio input status: {status}")
        if self.is_streaming and frames > 0:
            audio_bytes = in_data.tobytes()
            asyncio.run_coroutine_threadsafe(
                self.process_input_audio(audio_bytes),
                self.loop
            )

    async def process_input_audio(self, audio_data):
        """Process a single audio chunk directly"""
        try:
            # Send audio to Bedrock immediately
            self.stream_manager.add_audio_chunk(audio_data)
        except Exception as e:
            if self.is_streaming:
                print(f"Error processing input audio: {e}")
    
    async def play_output_audio(self):
        """Play audio responses from Nova Sonic"""
        while self.is_streaming:
            try:
                # Check for barge-in flag
                if self.stream_manager.barge_in:
                    # Clear the audio queue
                    while not self.stream_manager.audio_output_queue.empty():
                        try:
                            self.stream_manager.audio_output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    self.stream_manager.barge_in = False
                    # Small sleep after clearing
                    await asyncio.sleep(0.05)
                    continue
                
                # Get audio data from the stream manager's queue
                audio_data = await asyncio.wait_for(
                    self.stream_manager.audio_output_queue.get(),
                    timeout=0.1
                )
                
                if audio_data and self.is_streaming:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    if CHANNELS > 1:
                        audio_array = audio_array.reshape(-1, CHANNELS)

                    # Write the audio data in manageable chunks to avoid blocking
                    for start in range(0, audio_array.shape[0], CHUNK_SIZE):
                        if not self.is_streaming:
                            break

                        frame_chunk = audio_array[start:start + CHUNK_SIZE]

                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            self.output_stream.write,
                            frame_chunk
                        )

                        # Brief yield to allow other tasks to run
                        await asyncio.sleep(0.001)
                    
            except asyncio.TimeoutError:
                # No data available within timeout, just continue
                continue
            except Exception as e:
                if self.is_streaming:
                    print(f"Error playing output audio: {str(e)}")
                    import traceback
                    traceback.print_exc()
                await asyncio.sleep(0.05)
    
    async def start_streaming(self):
        """Start streaming audio."""
        if self.is_streaming:
            return
        
        print("Starting audio streaming. Speak into your microphone...")
        print("Press Enter to stop streaming...")
        
        self.is_streaming = True

        # Start the input stream if not already started
        if not self.input_stream.active:
            self.input_stream.start()
        if not self.output_stream.active:
            self.output_stream.start()
        
        # Start processing tasks
        #self.input_task = asyncio.create_task(self.process_input_audio())
        self.output_task = asyncio.create_task(self.play_output_audio())
        
        # Wait for user to press Enter to stop
        await asyncio.get_event_loop().run_in_executor(None, input)
        
        # Once input() returns, stop streaming
        await self.stop_streaming()
    
    async def stop_streaming(self):
        """Stop streaming audio."""
        if not self.is_streaming:
            return
            
        self.is_streaming = False

        # Cancel the tasks
        tasks = []
        if hasattr(self, 'input_task') and not self.input_task.done():
            tasks.append(self.input_task)
        if hasattr(self, 'output_task') and not self.output_task.done():
            tasks.append(self.output_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Stop and close the streams
        if self.input_stream:
            if self.input_stream.active:
                self.input_stream.stop()
            self.input_stream.close()
        if self.output_stream:
            if self.output_stream.active:
                self.output_stream.stop()
            self.output_stream.close()
        
        await self.stream_manager.close() 


async def main(debug=False):
    """Main function to run the application."""
    global DEBUG
    DEBUG = debug

    # Create stream manager
    stream_manager = BedrockStreamManager(model_id='amazon.nova-sonic-v1:0', region='us-east-1')

    # Create audio streamer
    audio_streamer = AudioStreamer(stream_manager)

    # Initialize the stream
    await time_it_async("initialize_stream", stream_manager.initialize_stream)

    try:
        # This will run until the user presses Enter
        await audio_streamer.start_streaming()
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        await audio_streamer.stop_streaming()
        

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Nova Sonic Python Streaming')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    # Set your AWS credentials here or use environment variables
    # os.environ['AWS_ACCESS_KEY_ID'] = "AWS_ACCESS_KEY_ID"
    # os.environ['AWS_SECRET_ACCESS_KEY'] = "AWS_SECRET_ACCESS_KEY"
    # os.environ['AWS_DEFAULT_REGION'] = "us-east-1"

    # Run the main function
    try:
        asyncio.run(main(debug=args.debug))
    except Exception as e:
        print(f"Application error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
