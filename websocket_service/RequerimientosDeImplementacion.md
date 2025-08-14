# Requerimientos de implementación

Analizar el script `realtime_microphone_vad.py` para crear un **servicio WebSocket** cuyo punto de entrada sea el archivo `websocket_service/server/vad_websocket_server.py`. El servicio debe permitir comunicación en tiempo real, transmitir audio **como streams** (no subir `.wav` entero), soportar **dos modos de streaming** y manejar múltiples clientes concurrentes, cada uno con su propia configuración de VAD y timeout.

> **Fundamental (regla invariable):** **NO** se debe cargar ni subir el archivo `.wav` como archivo entero por WebSocket. Cualquier audio (incluido `examples/audios/SampleVoiceMono.wav`) **debe** ser enviado como **PCM sin compresión** (raw PCM bytes) o como **paquetes comprimidos** (Opus o AAC) según el modo seleccionado. Queda prohibido enviar `.wav` entero como multipart/form-data o similar.

---

## Archivos / estructura de carpetas

* `examples/audios/SampleVoiceMono.wav` ← audio de prueba (17 s, voz humana, se espera detectar 4 segmentos)
* `websocket_service/`
  * `server/vad_websocket_server.py` ← punto de entrada del servidor WebSocket
  * `clients/python/`                ← cliente Python (soporta PCM y Opus/AAC)
  * `clients/java/`                  ← cliente Java (véase restricción abajo)
  * resto del código fuente dentro de `websocket_service` y subcarpetas únicamente

---

## Requisitos técnicos del servidor

1. **Frameworks / librerías**

   * Implementar servidor con **FastAPI**, **Pydantic**, y levantar con **uvicorn**.
   * Soporte de WebSocket/ASGI con Starlette (FastAPI lo provee).
   * Para decodificar Opus/AAC en el servidor usar librerías adicionales (ej. `av` / `PyAV` o wrappers a `libopus`) — dichas dependencias deben listarse en `pyproject.toml` y justificarse.

2. **Concurrencia**

   * El servidor debe soportar múltiples clientes concurrentes, con **estado aislado por cliente** (VAD config, timeout, stream-mode, buffers, contadores).
   * Todas las operaciones I/O deben ser `async` y no bloquear el loop principal.

3. **Configuración por cliente**

   * El cliente puede establecer parámetros por **query string** al conectarse (ejemplo más abajo) **y/o** puede enviar un mensaje JSON inicial de tipo `CONFIG` al abrir la conexión. Si ambos existen, la prioridad será: **mensaje CONFIG (JSON)** sobre **query params**. Si no se envía nada, usar valores por defecto.
   * Parámetros VAD por cliente:

     * `start_probability` (float, default 0.4)
     * `end_probability` (float, default 0.3)
     * `start_frame_count` (int, default 6)
     * `end_frame_count` (int, default 12)

   * Parámetros audio por cliente:

     * `mode`: `"pcm"` | `"opus"` | `"aac"` (default `"pcm"`)
     * `sample_rate` (int, default 16000) — **nota**: para Opus es común 48000; si el cliente elige `opus` y envía 48000, el servidor debe aceptarlo.
     * `channels` (int, default 1)
     * `sample_width` (bytes per sample, int, default 2) — normalmente 2 = 16-bit PCM, little-endian
     * `frame_duration_ms` (int, default 30) — duración del frame que el servidor espera para VAD (ver fórmula abajo)
    
    > **Nota**: Comprobar `realtime_microphone_vad.py` ratificando los valores usados en el VAD original.

   * `timeout` (float, segundos, default 0) — 0 significa *sin timeout*. Si >0, servidor debe enviar evento `TIMEOUT` cuando no haya voz por ese intervalo; el timeout se **reinicia** cada vez que se detecta voz.


4. **URL de conexión (ejemplo)**

```
ws://localhost:8000/vad?mode=pcm&sample_rate=16000&channels=1&sample_width=2&frame_duration_ms=30&start_probability=0.4&end_probability=0.3&start_frame_count=6&end_frame_count=12&timeout=2.0
```

* Los parámetros son opcionales y pueden venir en cualquier orden.

5. **Protocolo de mensajes**

   * **Mensajes de control (texto JSON)**:

     * `CONFIG` — cliente → servidor: configura/actualiza parámetros (VAD, audio, timeout). Ejemplo:

       ```json
       {
         "type": "CONFIG",
         "mode": "pcm",
         "sample_rate": 16000,
         "channels": 1,
         "sample_width": 2,
         "frame_duration_ms": 30,
         "start_probability": 0.4,
         "end_probability": 0.3,
         "start_frame_count": 6,
         "end_frame_count": 12,
         "timeout": 2.0
       }
       ```
     * `HEARTBEAT` (opcional) — cliente → servidor para mantener conexión viva.
   * **Mensajes de audio (binary WebSocket frames)**:

     * En **modo `pcm`** cada mensaje binario **debe** contener **exactamente un frame de PCM raw** cuya longitud en bytes sea:

       ```
       bytes_per_frame = sample_rate * (frame_duration_ms / 1000) * channels * sample_width
       ```

       * Ejemplo: `sample_rate=16000, frame_duration_ms=30, channels=1, sample_width=2` → `16000*0.03*1*2 = 960 bytes` por frame.
       * Cada mensaje binario corresponde a un frame (no envolver en JSON ni hacer base64).
     * En **modo `opus` o `aac`** cada mensaje binario **debe** contener exactamente **un paquete/paquete codificado** (tal como lo produce el encoder). El servidor recibirá esos paquetes y **si necesita** decodificarlos para alimentar el VAD, deberá hacerlo (ver dependencias).
     * **No se debe** enviar el `.wav` entero como un único binario; el almacenamiento local `StoredAudioSource` deberá **leer el .wav y enviar sus datos como frames PCM** conforme a la fórmula anterior.
   * **Eventos del servidor (texto JSON → servidor → cliente)**:

     * `VOICE_START`:

       ```json
       {"event":"VOICE_START","timestamp_ms":123456789,"segment_index":0}
       ```

     * `VOICE_CONTINUE`:

       ```json
       {"event":"VOICE_CONTINUE","timestamp_ms":123456889,"segment_index":0}
       ```

     * `VOICE_END`:

       ```json
       {"event":"VOICE_END","timestamp_ms":123456999,"segment_index":0,"segment_start_ms":123456789,"segment_end_ms":123456999,"duration_ms":210}
       ```

     * `TIMEOUT`:

       ```json
       {"event":"TIMEOUT","timestamp_ms":123457000,"message":"no voice detected in configured timeout"}
       ```

     * `ERROR` / `INFO` con estructura similar.
    
   * **Notas sobre timestamps / sincronización:** los timestamps pueden ser ms relativos al inicio de la conexión o epoch ms — el servidor debe documentar la elección y usarla consistentemente.

6. **VAD**

   * Implementación del VAD (usar la implementación del `VADConfig` original). 
   Debe consumir frames PCM (después de decodificar Opus/AAC si aplica).
   * La ventana de frames, conteos y probabilidades deben aplicarse por cliente. Reiniciar contadores al reconectar o al cambiar configuración.
   * Debe generar las notificaciones `VOICE_START`, `VOICE_CONTINUE` y `VOICE_END` en tiempo real y enviarlas al cliente que originó la señal.

7. **Timeout**

   * Default `timeout=0` (no timeout). **Las pruebas y ejemplos deben** usar `timeout > 0`. **No se permite** ejecutar pruebas sin timeout.
   * Si timeout > 0 y no se detecta voz en ese intervalo, enviar `TIMEOUT` al cliente. Reiniciar el timer cuando se detecte voz.

8. **Seguridad y robustez**

   * Validar con Pydantic todas las configuraciones y mensajes JSON.
   * Limitar tamaños máximos de frame y número de frames en buffer para evitar OOM o DoS.
   * Manejar desconexiones graciosas y liberar estado por cliente.

---

## Requisitos sobre los modos de streaming (PCM y Opus/AAC)

1. **Modo 1 — PCM sin compresión (modo `"pcm"`)**

   * Requisitos:

     * **El cliente** envía frames raw PCM (little-endian, signed integers si `sample_width`=2) como binary WebSocket messages, uno por frame.
     * El `StoredAudioSource` debe leer `examples/audios/SampleVoiceMono.wav`, extraer los samples PCM y enviar frames con la longitud correcta. **Nunca enviar el archivo .wav entero ni como campo multipart**.
     * El servidor procesa directamente esos bytes en el VAD sin decodificación adicional.
   * Ventaja: simple, sin pérdida ni latencia por encoding. Desventaja: más ancho de banda.

2. **Modo 2 — Audio en tiempo real con Opus o AAC (modo `"opus"` ó `"aac"`)**

   * Requisitos:

     * **El cliente Python** debe poder **codificar** PCM a paquetes Opus o AAC y enviar cada paquete como un binary WebSocket message (uno por paquete).
     * **El cliente Java** queda limitado a **modo PCM** bajo la condición "implementación nativa sin librerías adicionales al estándar para Java" (en la práctica **no existe un encoder Opus/AAC en la Java standard library**). Si se desea soporte Opus/AAC en Java, se deberá autorizar explícitamente el uso de librerías externas (p.ej. `libopus`/jni wrappers) — en este requerimiento actual **no** se autoriza; por ende Java sólo implementa PCM.
     * **El servidor** debe poder **decodificar** los paquetes Opus/AAC si es necesario para alimentar el VAD. Si no se dispone de decodificador en servidor, deberá rechazarse la conexión en modo `opus`/`aac` con un `ERROR` explicando la falta de soporte.
   * Dependencias (si se usa Opus/AAC):

     * Para Python: se autorizan las librerías mínimas necesarias (ej.: `av`/`PyAV`, `opuslib`, `soundfile` ó `pysoundfile`, `numpy`). Todos los paquetes deben declararse en `pyproject.toml`.
     * Para el servidor: si va a decodificar, incluir `av`/`ffmpeg` (nota: ffmpeg debe estar instalado en el sistema).
   * Formato de los paquetes: cada WebSocket binary debe contener exactamente un paquete codificado (no json ni base64). El servidor debe respetar los límites de paquetes y tiempos.

---

## Requisitos para los clientes (implementaciones y capas)

1. **2 clientes**: uno en **Python**, otro en **Java**.

2. **Carpetas**:

   * `websocket_service/clients/python/`
   * `websocket_service/clients/java/`

3. **Restricciones**:

   * **Java**: implementación nativa **sin librerías adicionales al estándar**. Esto implica que **el cliente Java sólo implementará el modo PCM** y `StoredAudioSource` debe extraer PCM del `.wav` y enviar frames PCM conforme a la especificación.
   * **Python**: usar sólo las librerías absolutamente necesarias. Está autorizado usar paquetes para manejo de audio y codificación (p.ej. `soundfile`, `numpy`, `av`/`PyAV`, `opuslib`) — cualquier dependencia adicional **debe** agregarse al `pyproject.toml`.

4. **Arquitectura interna (ambos clientes)**:

   * **Capa Source**:

     * `RealTimeMicSource`: captura audio del micrófono en tiempo real y emite frames PCM conforme al `frame_duration_ms`.
     * `StoredAudioSource`: lee `examples/audios/SampleVoiceMono.wav`, extrae PCM y emite frames PCM.
     * Ambos deben exponer los frames como streams de bytes listos para enviar (no encapsulados).
   
   * **Capa Gateway**:
     
     * Receibe streams de bytes desde cualquier `Source`.
     * `WebSocketGateway`: administra la conexión WebSocket, envía el mensaje `CONFIG` inicial (si procede), envía frames binarios al servidor y recibe eventos JSON.
     * Implementar reconexión simple y manejo de errores.
     * El Gateway debe aceptar `mode` y, si `mode` es `opus`/`aac` y el cliente soporta encodeo, aplicar el encoder antes de enviar.

5. **Pruebas con StoredAudioSource**

   * Crear un ejemplo en `websocket_service/examples/` que use **ambos clientes** Python y Java con `StoredAudioSource` enviando `examples/audios/SampleVoiceMono.wav`.
   * Iniciar las pruebas con el cliente Python hasta que se detecten los segmentos de voz esperados.
   * Luego de comprobar que el cliente Python funciona, probar y corregir el cliente Java con el mismo audio.
   * **Obligatorio**: las pruebas deben ejecutarse con `timeout > 0` configurable desde el cliente (no se permiten tests sin timeout).
   * Configuración esperada para la prueba:

     * `start_probability: 0.4`
     * `end_probability: 0.3`
     * `start_frame_count: 6`
     * `end_frame_count: 12`

     * `timeout: 30.0` (segundos)
    
   * **Verificación obligatoria**: comprobar que se detectan exactamente **4 segmentos de voz** y que los eventos `VOICE_START` / `VOICE_END` se correspondan a dichos segmentos. El test debe fallar si el conteo no coincide.
   * Los tests deben imprimir/registrar las métricas (timestamps, duración de cada segmento, número total de segmentos) y generar un exit code no-cero en caso de fallo.

---

## Reglas de diseño y buenas prácticas

* Usar **Pydantic** para todos los modelos de configuración y eventos.
* Mantener **una única fuente de verdad por cliente** (un objeto de contexto por conexión).
* Documentar claramente la **serialización** y **protocolos** en un README dentro de `websocket_service`.
* Registrar (log) los eventos importantes y errores con suficiente detalle (client\_id, timestamp, motivo).
* Validar tamaños de frames y rechazar conexiones con parámetros inconsistentes (p.ej. frame\_duration\_ms que no produce bytes enteros).
* Especificar claramente en el README que **los formatos de audio de ejemplo se leerán en PCM y serán enviados como frames**, nunca se subirán como archivos.

---

## Comandos de arranque y ejemplo de uso (debe documentarse e incluirse en README)

* Iniciar servidor:

```
uvicorn websocket_service.server.vad_websocket_server:app --host 0.0.0.0 --port 8000
```

* Conectar cliente (ejemplo con query params):

```
ws://localhost:8000/vad?mode=pcm&sample_rate=16000&channels=1&frame_duration_ms=30&start_probability=0.4&end_probability=0.3&start_frame_count=6&end_frame_count=12&timeout=2.0
```

* Flujo típico:

  1. Cliente abre WS con query params.
  2. Cliente envía `CONFIG` JSON (opcional).
  3. Cliente envía frames binarios (uno por mensaje).
  4. Servidor procesa frames, emite eventos `VOICE_START` / `VOICE_CONTINUE` / `VOICE_END` / `TIMEOUT`.
  5. Cliente visualiza/afirma resultados; test automatizado valida que se detectaron 4 segmentos con la configuración de prueba.

---

## Mensajes de salida / eventos esperados en la prueba

* Durante la prueba con `SampleVoiceMono.wav`, el cliente (o el test runner) debe recibir y registrar al menos los siguientes eventos en orden y con coherencia temporal:

  1. `VOICE_START` (segment\_index 0)
  2. `VOICE_END`   (segment\_index 0)
  3. `VOICE_START` (segment\_index 1)
  4. `VOICE_END`   (segment\_index 1)
  5. `VOICE_START` (segment\_index 2)
  6. `VOICE_END`   (segment\_index 2)
  7. `VOICE_START` (segment\_index 3)
  8. `VOICE_END`   (segment\_index 3)
* Además, si se configura `timeout > 0` y ocurre, debe recibirse `TIMEOUT`. Las pruebas obligatoriamente deben configurar `timeout` para evitar bloqueos indefinidos.

---

Implementa el servidor y clientes siguiendo estrictamente estos requerimientos y coloca todo el código fuente únicamente dentro de `websocket_service` o `examples` y sus subcarpetas.

Debes hacer tu razonamiento y respuestas en inglés.
