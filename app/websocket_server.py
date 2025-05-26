import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import logging
from .lsf_recognizer import recognize_lsf_sign

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('LSF_Server')

class LSFWebSocketServer:
    def __init__(self):
        self.clients = set()
        logger.info("Serveur LSF initialisé")

    async def register(self, websocket):
        self.clients.add(websocket)
        logger.info(f"Nouvelle connexion WebSocket. Clients connectés : {len(self.clients)}")
        # Envoie le message de connexion à chaque nouveau client
        await websocket.send(json.dumps({"type": "connection_established"}))

    async def unregister(self, websocket):
        self.clients.remove(websocket)
        logger.info(f"Client déconnecté. Clients connectés : {len(self.clients)}")

    def fix_base64_padding(self, b64_string):
        return b64_string + '=' * (-len(b64_string) % 4)

    async def handle_client(self, websocket):
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    # Log the received message for debugging
                    logger.info(f"Message received: {message[:200]}...") # Log first 200 chars
                    # Décoder le message JSON
                    data = json.loads(message)
                    if data.get("type") == "image":
                        image_b64 = data.get("data")
                        if not image_b64:
                            raise ValueError("Champ 'data' manquant ou vide dans le message JSON.")
                        # Corriger le padding base64 si besoin
                        image_b64 = self.fix_base64_padding(image_b64)
                        image_data = base64.b64decode(image_b64)
                        nparr = np.frombuffer(image_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is None:
                            raise ValueError("Impossible de décoder l'image envoyée.")
                        # Reconnaître le signe
                        sign = recognize_lsf_sign(frame)
                        # Envoyer la réponse
                        response = {
                            "type": "sign_detected",
                            "sign": sign
                        }
                        await websocket.send(json.dumps(response))
                        logger.info(f"Signe détecté et envoyé : {sign}")
                    else:
                        # Message non reconnu
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"Type de message non supporté : {data.get('type')}"
                        }))
                except json.JSONDecodeError as e:
                    logger.error(f"Erreur de décodage JSON : {str(e)}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Format JSON invalide"
                    }))
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de l'image : {str(e)}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connexion WebSocket fermée")
        finally:
            await self.unregister(websocket)

async def start_server():
    server = LSFWebSocketServer()
    try:
        async with websockets.serve(server.handle_client, "0.0.0.0", 8765):
            logger.info("Serveur LSF démarré sur ws://0.0.0.0:8765")
            await asyncio.Future()  # Garde le serveur en vie
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du serveur : {str(e)}")
        raise

def start_lsf_server():
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("Arrêt du serveur LSF")
    except Exception as e:
        logger.error(f"Erreur fatale : {str(e)}")
        raise 