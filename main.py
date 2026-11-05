"""
main.py - Script de prueba del entregable (el que se ejecuta con: python main.py).

Mapa de la consigna dentro de este archivo:
    PUNTO 2 -> se construye GenerationParams y se captura ValidationError.
    PUNTO 3 -> se crea AsyncLLMManager() sin argumentos: el proveedor sale de .env.
    PUNTO 4 -> se consume el stream con "async for" e imprime cada fragmento.
    PUNTO 5 -> se distingue un LLMError (error controlado) del texto normal.
    PUNTO 6 -> el README.md explica como ejecutar este script.
"""

# asyncio es la libreria estandar para ejecutar codigo asincrono (async/await).
import asyncio
import sys

from pydantic import ValidationError

from llm_manager import AsyncLLMManager, GenerationParams, LLMError


async def main() -> None:
    try:
        # PUNTO 2 - Se pide texto al usuario y se crea el objeto validado por
        # Pydantic. Si el prompt queda vacio o temperature/max_tokens estan
        # fuera de rango, aca mismo se lanza ValidationError.
        params = GenerationParams(
            prompt=input("Escribe tu prompt: ").strip(),
            system="Eres un asistente util.",
            temperature=0.7,
            max_tokens=500,
        )

        # PUNTO 3 - Se crea el gestor. Lee LLM_PROVIDER y la clave desde .env.
        # Para cambiar de proveedor no se toca codigo: solo el archivo .env.
        manager = AsyncLLMManager()

        # PUNTO 4 - Se reciben fragmentos y se escriben de inmediato.
        #   end=""      -> no agrega salto de linea entre fragmentos.
        #   flush=True  -> fuerza a mostrar el texto ya, sin esperar al buffer.
        # Asi se ve el efecto "maquina de escribir" del streaming.
        async for fragment in manager.generate(params):
            # PUNTO 5 - generate() entrega textos (str) o, si algo fallo, un
            # LLMError. Se distinguen por su TIPO, no leyendo el texto.
            if isinstance(fragment, LLMError):
                # El error va a stderr; fragment.kind permitiria reaccionar
                # distinto segun el caso (ej. un agente esperando si es RATE_LIMIT).
                print(f"\n{fragment}", file=sys.stderr)
            else:
                print(fragment, end="", flush=True)
        print()  # salto de linea final cuando termina la respuesta

    except ValidationError as exc:
        # PUNTO 2 - Error de Pydantic: prompt, temperatura o max_tokens invalidos.
        # Va ANTES que ValueError porque ValidationError hereda de ValueError;
        # si estuviera despues, nunca se ejecutaria este bloque.
        print(f"Parametros invalidos: {exc}")
    except ValueError as exc:
        # PUNTO 3 - LLM_PROVIDER invalido o clave de API faltante.
        print(f"Configuracion invalida: {exc}")
    except KeyboardInterrupt:
        # El usuario presiono Ctrl+C.
        print("\nGeneracion cancelada.")


# Este bloque solo corre cuando se ejecuta "python main.py" directamente,
# no cuando otro archivo hace "import main".
# asyncio.run() crea el event loop, ejecuta main() y lo cierra al terminar.
if __name__ == "__main__":
    asyncio.run(main())
