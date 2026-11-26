"""Simulador do totem de verificação de idade (sistema embarcado).

Representa um dispositivo de borda (ex.: câmera com processador embarcado) instalado num ponto de
cadastro. A cada captura, o dispositivo:
  1. detecta e recorta o rosto;
  2. calcula o embedding FaRL (512 números) no próprio dispositivo;
  3. publica SÓ o embedding via MQTT, nunca a foto (privacidade desde a concepção, LGPD art. 46).

Na simulação, os passos 1 e 2 já foram feitos no notebook 01: o totem lê embeddings reais de imagens
que o modelo nunca viu no treino (split "test" do UTKFace e o FG-NET) e os publica em tempo real.

Tópico:  veridade/totem/<totem_id>/captura      QoS 1
Payload: {"captura_id", "totem_id", "capturado_em", "embedding": [512], "idade_real", "origem"}
"""
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

HOST = os.getenv("MQTT_HOST", "localhost")
PORT = int(os.getenv("MQTT_PORT", "1883"))
TOTENS = os.getenv("TOTENS", "totem-recife-01").split(",")
INTERVALO = float(os.getenv("INTERVALO_S", "1.0"))
ARQUIVO = os.getenv("FEATURES", "data/processed/features_farl.parquet")
LIMITE = int(os.getenv("LIMITE_CAPTURAS", "0"))  # 0 = infinito (recomeça a lista ao final)
SEMENTE = int(os.getenv("SEMENTE", "42"))


def carregar_capturas() -> pd.DataFrame:
    df = pd.read_parquet(ARQUIVO)
    df = df[df["split"].isin(["test", "test_externo"])].reset_index(drop=True)
    print(f"[totem] {len(df)} capturas disponíveis (UTKFace teste + FG-NET)", flush=True)
    return df


def conectar() -> mqtt.Client:
    cliente = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"totem-sim-{uuid.uuid4().hex[:6]}")
    for tentativa in range(30):
        try:
            cliente.connect(HOST, PORT, keepalive=30)
            cliente.loop_start()
            print(f"[totem] conectado ao broker {HOST}:{PORT}", flush=True)
            return cliente
        except OSError:
            print(f"[totem] broker indisponível, tentando de novo ({tentativa + 1}/30)...", flush=True)
            time.sleep(2)
    raise SystemExit("[totem] não foi possível conectar ao broker MQTT")


def main():
    df = carregar_capturas()
    emb_cols = [f"e{i}" for i in range(512)]
    E = df[emb_cols].to_numpy(np.float32)
    rng = random.Random(SEMENTE)
    ordem = list(range(len(df)))
    cliente = conectar()

    enviadas, i = 0, 0
    while LIMITE == 0 or enviadas < LIMITE:
        if i % len(ordem) == 0:
            rng.shuffle(ordem)
        idx = ordem[i % len(ordem)]; i += 1
        linha = df.iloc[idx]
        totem = rng.choice(TOTENS)
        payload = {
            "captura_id": uuid.uuid4().hex,
            "totem_id": totem,
            "capturado_em": datetime.now(timezone.utc).isoformat(),
            "embedding": [round(float(x), 6) for x in E[idx]],
            "idade_real": int(linha["idade"]),
            "origem": "FG-NET" if linha["dataset"] == "FG-NET" else "UTKFace-teste",
        }
        info = cliente.publish(f"veridade/totem/{totem}/captura", json.dumps(payload), qos=1)
        info.wait_for_publish(timeout=5)
        enviadas += 1
        if enviadas % 30 == 0:
            print(f"[totem] {enviadas} capturas publicadas", flush=True)
        time.sleep(INTERVALO)

    cliente.loop_stop(); cliente.disconnect()


if __name__ == "__main__":
    main()
