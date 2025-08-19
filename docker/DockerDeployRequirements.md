Analiza detenidamente, el servidor websocket determina sus dependencias para crear un archivo `docker-compose.yml` que permita levantar el servidor en un ambiente aislado, con las dependencias necesarias para su funcionamiento. Deben aplicarse las mejores prácticas para un despliegue listo para producción, incluyendo la configuración de variables de entorno, puertos y volúmenes si es necesario.
Si se requieren imágenes personalizadas utiliza como base una imagen oficial de alpine python o similar, asegurando que el entorno sea ligero y eficiente.

Es requisito indispensable mantener la configuración, la estructura y el despliegue del servidor websocket con docker tan simples y claros como sea posible.

Actualmente el servidor websocket se ejecuta con el comando:
```bash
.\venv\Scripts\activate
uvicorn websocket_service.server.vad_websocket_server:app --host 0.0.0.0 --port 8000 --log-level debug
```

Y se ha comprobado que funciona correctamente con la siguiente prueba en python:
```bash
.\venv\Scripts\activate
python examples/test_python_vad_client.py
```

Crea toda la implementación necesaria en la carpeta `docker`

Testear completamente el despliegue con docker usando el test `examples/test_python_vad_client.py` para verificar que el servidor websocket funciona correctamente con el audio de ejemplo `examples/audios/SampleVoice.wav` y la configuración de prueba mencionada anteriormente, y corregir cualquier fallo que se presente.

No puedes modificar el archivo `examples/test_python_vad_client.py`, debes asegurarte de que el servidor websocket funcione correctamente con la configuración actual y el audio de ejemplo.

Se sabe que con la configuración de la prueba y el audio de ejemplo `examples/audios/SampleVoice.wav` se deben detectar cuatro segmentos de voz.

Debes hacer tu razonamiento y respuestas en inglés.