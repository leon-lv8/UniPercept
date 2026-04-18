docker compose --profile download run  hf-download

docker compose --profile gpu up --build

docker system prune -a




bash docker/entrypoint.sh



conda activate unipercept

bash ./src//eval/conversation.sh

