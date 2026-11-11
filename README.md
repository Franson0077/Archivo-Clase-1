# Curso de agentes: streaming con LLMs

Proyecto de ejemplo para configurar modelos OpenAI o Anthropic de forma asincrona.

## Estructura del proyecto

```text
schemas.py        -> QUE datos entran y salen (GenerationParams, LLMError)
providers.py      -> COMO se habla con cada API (BaseProvider + una clase por proveedor)
llm_manager.py    -> QUE proveedor usar y QUE hacer si falla (AsyncLLMManager)
main.py           -> script de prueba interactivo
test_fallback.py  -> pruebas automaticas sin gastar tokens
```

Cada archivo depende solo de los de arriba: `schemas.py` no conoce a nadie,
`providers.py` usa `schemas.py`, y `llm_manager.py` usa a los dos.

## Mapa de la consigna

| Punto | Consigna | Dónde está |
|---|---|---|
| 1 | Entorno virtual + openai, anthropic, pydantic, python-dotenv | `requirements.txt`, sección *Instalacion* |
| 2 | Esquema Pydantic (temperatura 0–2, max_tokens, etc.) | `schemas.py` → `GenerationParams` |
| 3 | `AsyncLLMManager` que carga OpenAI o Anthropic según configuración | `llm_manager.py` → `AsyncLLMManager.__init__` + `LLM_PROVIDER`; `providers.py` → `PROVIDERS` |
| 4 | Streaming con `yield` | `providers.py` → `stream()` de cada proveedor; `llm_manager.py` → `generate` |
| 5 | Errores de red y rate limit controlados | `schemas.py` → `LLMError`; `providers.py` → `classify_error`; `llm_manager.py` → `generate` |
| 6 | README con ejecución y variables de entorno | este archivo |
| Prueba | Demostración de los puntos 2 y 5 sin gastar tokens | `test_fallback.py` |

En el código cada parte está marcada con un comentario `PUNTO N`.

## Requisitos

- Python 3.15 (la consigna menciona 3.12; se eligió 3.15 a propósito, el
  código no usa nada específico de una versión y funciona desde 3.10)
- Una clave de API para OpenAI o Anthropic

## Instalacion

Desde la carpeta del proyecto:

```powershell
py -3.15 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Error "la ejecución de scripts está deshabilitada en este sistema"** al
> activar el entorno: es una protección de PowerShell. Habilitalo solo para la
> terminal actual y volvé a activar:
>
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .venv\Scripts\activate
> ```
>
> Para no repetirlo cada vez, usá `-Scope CurrentUser -ExecutionPolicy RemoteSigned`.
> También se puede evitar activar el entorno y usar directamente
> `.venv\Scripts\python.exe main.py`.

## Configuracion

Copia `.env.example` como `.env` y completa los valores necesarios:

```dotenv
LLM_PROVIDER=anthropic
LLM_FALLBACK_PROVIDER=openai

ANTHROPIC_API_KEY=tu_clave
ANTHROPIC_MODEL=claude-sonnet-4-5

OPENAI_API_KEY=tu_clave
OPENAI_MODEL=gpt-4o-mini
```

Variables disponibles:

- `LLM_PROVIDER`: proveedor principal, `openai` o `anthropic` (por defecto `anthropic`).
- `LLM_FALLBACK_PROVIDER`: proveedor de respaldo (opcional). Si se deja vacio,
  no hay respaldo y solo se necesita la clave del principal.
- `ANTHROPIC_API_KEY`: clave para Anthropic.
- `ANTHROPIC_MODEL`: modelo de Anthropic; por defecto `claude-sonnet-4-5`.
- `OPENAI_API_KEY`: clave para OpenAI.
- `OPENAI_MODEL`: modelo de OpenAI; por defecto `gpt-4o-mini`.
- `LLM_MAX_RETRIES` (opcional): reintentos automáticos del SDK ante 429, 5xx o
  errores de red, antes de pasar al respaldo. Por defecto `2`. Con `0` el
  respaldo entra de inmediato.
- `LLM_TIMEOUT` (opcional): segundos máximos de espera por petición. Por
  defecto `60`.

Si hay respaldo configurado, se necesitan **las dos claves** y se comprueban
al iniciar: asi no te enteras de que falta una justo cuando el principal se cae.

## Seguridad de las claves

No pongas claves reales ni dentro de `.env.example` ni directamente en el código.
Copia `.env.example` como `.env`; este archivo está excluido de Git mediante
`.gitignore`.

```powershell
Copy-Item .env.example .env
```

`Pydantic` valida los parámetros enviados al modelo, pero no protege las claves
ni es un sistema de permisos. Si la aplicación será pública, nunca expongas la
clave en JavaScript o en una aplicación de escritorio. Mantén las llamadas al
proveedor en un backend propio y entrega un token de sesión o autenticación al
cliente, sin incluir la clave de OpenAI o Anthropic en el código fuente.

## Ejecucion

```powershell
.venv\Scripts\activate
python main.py
```

El texto se imprime a medida que llega desde el proveedor.

## Pruebas sin gastar tokens

```powershell
python -m unittest -v test_fallback
```

`test_fallback.py` reemplaza las llamadas reales por funciones falsas que
simulan caídas (red, 429, 500, 529, clave inválida, corte a mitad de
respuesta). No necesita internet ni claves reales, y usa `unittest`, que viene
con Python, así que no hay que instalar nada extra. Comprueba que:

- Pydantic rechaza parámetros fuera de rango (punto 2).
- Cada error termina en un `LLMError` controlado, sin crash (punto 5).
- Si Anthropic se cae, se usa OpenAI con la `temperature` y el `top_p` validados.
- Se puede sumar un proveedor nuevo (`EchoProvider`, de juguete) sin tocar
  `llm_manager.py`.
- Un bug de nuestro propio código no se disfraza de error de la API.

## Respaldo automatico (Anthropic -> OpenAI)

Si el proveedor principal se cae, primero el SDK reintenta solo
(`LLM_MAX_RETRIES` veces, esperando un poco más cada vez). Si sigue sin
responder, `generate()` pasa al proveedor de respaldo y muestra un aviso:

```text
[AVISO] anthropic no disponible (SERVER). Reintentando con openai...
```

| Situacion | Que pasa |
|---|---|
| Limite de tasa (429) | Usa el respaldo |
| Error de red o timeout | Usa el respaldo |
| Error del servidor 5xx / saturado (529) | Usa el respaldo |
| Clave invalida, peticion mal armada | `[ERROR:PROVIDER]`, sin respaldo (es un error de configuracion) |
| Se cae a mitad de la respuesta | `[ERROR:...]`, sin respaldo (evita texto duplicado) |
| El respaldo tambien falla | `[ERROR:RATE_LIMIT]`, `[ERROR:NETWORK]` o `[ERROR:SERVER]` |

Cuando entra OpenAI se usan `temperature` y `top_p` tal como se validaron en
`GenerationParams`.

## Guia rapida del codigo

**`schemas.py`**

- `GenerationParams`: estructura Pydantic para validar `prompt`, `temperature`,
  `max_tokens`, `top_p` y `system`.
- `LLMError`: error controlado con `kind` (`RATE_LIMIT`, `NETWORK`, `SERVER`,
  `PROVIDER`), `provider` y `message`.

**`providers.py`**

- `BaseProvider`: clase base abstracta. Define lo que todo proveedor tiene que
  saber hacer: `create_client()`, `stream()` y `classify_error()`. También lee
  la clave y el modelo desde `.env`.
- `OpenAIProvider` y `AnthropicProvider`: implementan esos métodos con su SDK.
  `stream()` entrega fragmentos con `yield` y no captura errores: los deja subir
  al manager.
- `PROVIDERS`: diccionario `{"openai": OpenAIProvider, ...}`. Es el registro de
  proveedores que se pueden elegir desde `.env`.

**`llm_manager.py`**

- `AsyncLLMManager.__init__`: lee `LLM_PROVIDER`, `LLM_FALLBACK_PROVIDER`,
  `LLM_MAX_RETRIES` y `LLM_TIMEOUT`, busca cada proveedor en `PROVIDERS` y lo
  crea. No tiene ningún `if openai / else anthropic`.
- `AsyncLLMManager.generate`: recorre los proveedores en orden, reenvía los
  fragmentos mediante `yield` y decide si usar el respaldo. Entrega textos
  (`str`) y, si algo falla, un único `LLMError` al final.

**`main.py`**: pide el prompt, crea los parámetros, inicia el gestor y muestra
cada fragmento sin esperar la respuesta completa.

## Como agregar un proveedor nuevo

1. En `providers.py`, crear una clase que herede de `BaseProvider`:

   ```python
   class MistralProvider(BaseProvider):
       name = "mistral"
       api_key_env = "MISTRAL_API_KEY"
       model_env = "MISTRAL_MODEL"
       default_model = "mistral-small"
       rate_limit_errors = (...)   # errores de rate limit de su SDK
       network_errors = (...)      # errores de red
       server_errors = (...)       # errores 5xx
       api_error = ...             # clase madre de los errores del SDK

       def create_client(self, api_key, max_retries, timeout): ...
       async def stream(self, params): ...   # yield de cada fragmento
   ```

2. Registrarla en el diccionario `PROVIDERS` (una línea).
3. Usarla desde `.env`: `LLM_PROVIDER=mistral` o `LLM_FALLBACK_PROVIDER=mistral`.

No hace falta tocar `llm_manager.py` ni `main.py`. `test_fallback.py` lo
demuestra con `EchoProvider`, un proveedor de juguete.

## Que errores se controlan

El punto 5 está implementado en `LLMError` (`schemas.py`), `classify_error()`
(`providers.py`) y `generate()` (`llm_manager.py`). El error se devuelve como un
objeto, no como texto, para que el código que consume el stream (por ejemplo,
un agente) pueda reaccionar según el tipo:

```python
async for fragmento in manager.generate(params):
    if isinstance(fragmento, LLMError):
        print(fragmento.kind)   # "RATE_LIMIT", "NETWORK", "SERVER" o "PROVIDER"
    else:
        print(fragmento, end="")
```

| Error del SDK | `LLMError.kind` |
|---|---|
| `RateLimitError` | `RATE_LIMIT` |
| `APIConnectionError` / `APITimeoutError` | `NETWORK` |
| `InternalServerError` (incluye `OverloadedError` de Anthropic) | `SERVER` |
| Cualquier otro `APIError` (clave inválida, petición mal armada) | `PROVIDER` |

**Alcance.** El "error controlado en lugar de un crash" cubre los errores **de
la API** durante la generación. Hay dos casos que, a propósito, **no** se
convierten en `LLMError`:

| Caso | Qué pasa | Por qué |
|---|---|---|
| Falta una clave o `LLM_PROVIDER` es inválido | `ValueError` al crear `AsyncLLMManager()` | Es un error de configuración: se detecta al iniciar, antes de generar nada. `main.py` lo captura y muestra `Configuracion invalida: ...`. |
| Bug de nuestro código (ej. `AttributeError`) | Se propaga con su traceback | Disfrazarlo de error del proveedor escondería el problema real. Lo comprueba `test_un_bug_de_codigo_no_se_disfraza_de_error_de_api`. |

## Nota sobre Anthropic

La version de Anthropic instalada (2.5.0) no acepta `temperature` ni `top_p`.
Esos valores se validan igual con Pydantic porque los usa OpenAI cuando actua
como respaldo.
