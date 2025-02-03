from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables through config.py
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

AWS_ACCESS_KEY_ID_1 = os.getenv('AWS_ACCESS_KEY_ID_1')
AWS_SECRET_ACCESS_KEY_1 = os.getenv('AWS_SECRET_ACCESS_KEY_1')
AWS_REGION_1 = os.getenv('AWS_REGION_1')