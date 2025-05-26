from app.websocket_server import start_lsf_server
import base64

def fix_base64_padding(b64_string):
    return b64_string + '=' * (-len(b64_string) % 4)

if __name__ == "__main__":
    start_lsf_server() 