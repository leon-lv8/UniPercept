conda activate unipercept

docker compose --profile gpu up --build

bash ./src//eval/conversation.sh
