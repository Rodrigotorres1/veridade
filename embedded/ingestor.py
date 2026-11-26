"""Ingestor: assina o MQTT e grava cada captura no PostgreSQL com status "pendente".

Fica do lado do servidor (gateway). Não faz inferência: só garante que nenhuma captura se perca.
A inferência é feita pela DAG do Airflow, em lotes, o que permite reprocessar e monitorar.
"""
import json
import os
import time

import paho.mqtt.client as mqtt
import psycopg2

HOST = os.getenv("MQTT_HOST", "localhost")
PORT = int(os.getenv("MQTT_PORT", "1883"))
DB_URL = os.getenv("VERIDADE_DB_URL", "postgresql://veridade:veridade@localhost:5433/veridade")
TOPICO = "veridade/totem/+/captura"

SQL = """
INSERT INTO capturas (captura_id, totem_id, capturado_em, embedding, idade_real, origem)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (captura_id) DO NOTHING
"""


def conectar_db():
    for tentativa in range(30):
        try:
            conn = psycopg2.connect(DB_URL)
            conn.autocommit = True
            print("[ingestor] conectado ao PostgreSQL", flush=True)
            return conn
        except psycopg2.OperationalError:
            print(f"[ingestor] banco indisponível, tentando de novo ({tentativa + 1}/30)...", flush=True)
            time.sleep(2)
    raise SystemExit("[ingestor] não foi possível conectar ao banco")


def main():
    conn = conectar_db()
    contador = {"ok": 0, "erro": 0}

    def ao_conectar(cliente, userdata, flags, reason_code, properties):
        cliente.subscribe(TOPICO, qos=1)
        print(f"[ingestor] inscrito em {TOPICO}", flush=True)

    def ao_receber(cliente, userdata, msg):
        nonlocal conn
        try:
            p = json.loads(msg.payload)
            with conn.cursor() as cur:
                cur.execute(SQL, (p["captura_id"], p["totem_id"], p["capturado_em"], p["embedding"],
                                  p.get("idade_real"), p.get("origem")))
            contador["ok"] += 1
            if contador["ok"] % 30 == 0:
                print(f"[ingestor] {contador['ok']} capturas gravadas ({contador['erro']} com erro)", flush=True)
        except psycopg2.InterfaceError:
            conn = conectar_db()
        except Exception as e:  # mensagem malformada não derruba o serviço
            contador["erro"] += 1
            print(f"[ingestor] mensagem descartada: {e}", flush=True)

    cliente = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="veridade-ingestor")
    cliente.on_connect, cliente.on_message = ao_conectar, ao_receber
    for tentativa in range(30):
        try:
            cliente.connect(HOST, PORT, keepalive=30)
            break
        except OSError:
            print(f"[ingestor] broker indisponível ({tentativa + 1}/30)...", flush=True)
            time.sleep(2)
    cliente.loop_forever()


if __name__ == "__main__":
    main()
