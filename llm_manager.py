"""
llm_manager.py - Modulo principal del entregable.

Mapa de la consigna dentro de este archivo:
    PUNTO 1 -> imports de openai, anthropic, pydantic y python-dotenv
               (se instalan desde requirements.txt dentro del .venv).
    PUNTO 2 -> clase GenerationParams (esquema Pydantic de validacion).
    PUNTO 3 -> clase AsyncLLMManager, metodo __init__ (elige proveedor segun
               la variable de configuracion LLM_PROVIDER).
    PUNTO 4 -> metodos generate, _generate_openai y _generate_anthropic
               (streaming con yield).
    PUNTO 5 -> clase LLMError + metodo generate + _classify_error
               (rate limit y errores de red devueltos como error controlado).

Extras (no pedidos por la consigna):
    - Proveedor de RESPALDO: si el principal (ej. Anthropic) se cae,
      generate() reintenta con LLM_FALLBACK_PROVIDER (ej. OpenAI), usando la
      temperatura y top_p validados por Pydantic.
    - Reintentos y timeout configurables (LLM_MAX_RETRIES, LLM_TIMEOUT).
"""

import os
import sys

# AsyncIterator: tipo para "algo que se recorre con async for".
# Literal: restringe un valor a opciones exactas ("openai" o "anthropic").
from typing import AsyncIterator, Literal

# PUNTO 1 - Dependencias pedidas por la consigna (python-dotenv y pydantic).
# load_dotenv() busca un archivo .env y carga sus pares clave=valor en os.environ.
# Asi las claves quedan fuera de llm_manager.py y pueden cambiar sin modificar codigo.
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# PUNTO 1 - Dependencias pedidas por la consigna (anthropic y openai).
# Los SDK son diferentes, por eso se importan dos clientes y dos familias de errores.
# Se renombran con "as" porque ambos SDK usan los mismos nombres de clase
# (RateLimitError, APIConnectionError...) y sin alias chocarian entre si.
from anthropic import AsyncAnthropic
from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APIError as AnthropicAPIError
from anthropic import APITimeoutError as AnthropicTimeoutError
from anthropic import InternalServerError as AnthropicServerError
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import AsyncOpenAI
from openai import APIConnectionError as OpenAIConnectionError
from openai import APIError as OpenAIAPIError
from openai import APITimeoutError as OpenAITimeoutError
from openai import InternalServerError as OpenAIServerError
from openai import RateLimitError as OpenAIRateLimitError


# Se ejecuta una sola vez, al importar el modulo. Desde aca en adelante
# os.getenv("...") ya puede leer las variables definidas en .env.
load_dotenv()


Provider = Literal["openai", "anthropic"]
VALID_PROVIDERS = {"openai", "anthropic"}

# PUNTO 5 - Agrupamos las clases de error por categoria. Son tuplas porque
# "except (A, B, C)" e "isinstance(exc, (A, B, C))" aceptan tuplas.
RATE_LIMIT_ERRORS = (AnthropicRateLimitError, OpenAIRateLimitError)
NETWORK_ERRORS = (
    AnthropicConnectionError,
    AnthropicTimeoutError,
    OpenAIConnectionError,
    OpenAITimeoutError,
)
# Errores 5xx del servidor. En Anthropic incluye OverloadedError (529,
# "servidor saturado"), que es subclase de InternalServerError.
SERVER_ERRORS = (AnthropicServerError, OpenAIServerError)

# Errores "de caida": el proveedor no esta disponible en este momento.
# Solo estos activan el respaldo.
RECOVERABLE_ERRORS = RATE_LIMIT_ERRORS + NETWORK_ERRORS + SERVER_ERRORS

# Cualquier otro error que venga de la API (clave invalida, peticion mal
# armada, modelo inexistente...). APIError es la clase madre de TODOS los
# errores de cada SDK, por eso se captura al final como "otro error de API".
API_ERRORS = (AnthropicAPIError, OpenAIAPIError)


# =============================================================================
# PUNTO 2 - Esquema Pydantic para validar los parametros de entrada del modelo.
# =============================================================================
class GenerationParams(BaseModel):
    # Pydantic valida los datos cuando se crea GenerationParams(...).
    # Si un dato no cumple, se lanza una ValidationError ANTES de llamar al
    # proveedor: no se gasta una peticion (ni dinero) con datos invalidos.
    #
    # Field(...) declara las reglas de cada campo:
    #   ge = mayor o igual   | gt = mayor estricto
    #   le = menor o igual   | min_length = largo minimo del texto

    # Texto que escribe el usuario. Obligatorio (no tiene default).
    prompt: str = Field(min_length=1, description="Texto enviado al modelo")

    # Instruccion de sistema: define el "rol" o comportamiento del modelo.
    system: str | None = Field(default=None, description="Instruccion de sistema opcional")

    # PUNTO 2 - "temperatura de 0 a 2": ge=0.0 y le=2.0 cumplen exactamente eso.
    # Valores bajos = respuestas mas deterministas; altos = mas creativas.
    # IMPORTANTE: solo la usa OpenAI. El SDK de Anthropic instalado no acepta
    # temperature, pero se valida igual porque si Anthropic se cae, el
    # respaldo (OpenAI) la necesita.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # PUNTO 2 - "max_tokens": limite de largo de la respuesta. Debe ser > 0.
    # Este si lo usan ambos proveedores.
    max_tokens: int = Field(default=500, gt=0, le=100000)

    # Parametro extra (el "etc." de la consigna): nucleus sampling, entre 0 y 1.
    # Igual que temperature: solo lo usa OpenAI.
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)

    # Validadores personalizados: corren DESPUES de las reglas de Field.
    # min_length=1 acepta "   " (tres espacios), por eso hace falta este
    # validador extra que limpia espacios y rechaza el texto vacio.
    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("El prompt no puede estar vacio.")
        return value

    # Si system llega vacio o con solo espacios, se normaliza a None
    # para que mas adelante no se envie un mensaje de sistema vacio.
    @field_validator("system")
    @classmethod
    def validate_system(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None


# =============================================================================
# PUNTO 5 - Error controlado: un OBJETO, no un texto suelto.
# =============================================================================
class LLMError(BaseModel):
    # Antes el error llegaba como un string mas del stream ("[ERROR:...] ..."),
    # y quien lo recibia no podia distinguirlo de una respuesta del modelo.
    # Ahora generate() entrega textos (str) o, si algo falla, UN LLMError.
    # Quien consume pregunta: isinstance(fragmento, LLMError).
    #
    # Para un AGENTE esto es clave: puede decidir que hacer segun el tipo de
    # error (esperar, avisar, cambiar de herramienta...) en vez de solo mostrarlo.
    kind: Literal["RATE_LIMIT", "NETWORK", "SERVER", "PROVIDER"]
    provider: str
    message: str

    def __str__(self) -> str:
        # Formato legible para mostrar en la terminal.
        return f"[ERROR:{self.kind}] {self.provider}: {self.message}"


# =============================================================================
# PUNTO 3 - Clase AsyncLLMManager: carga OpenAI o Anthropic segun configuracion.
# =============================================================================
class AsyncLLMManager:
    # Esta clase prepara un proveedor principal y, opcionalmente, uno de respaldo.
    # El resto del programa (main.py) no necesita saber que proveedor se usa:
    # siempre llama a manager.generate(params) y recibe texto o un LLMError.
    def __init__(
        self,
        provider: Provider | None = None,
        fallback_provider: Provider | None = None,
        openai_model: str | None = None,
        anthropic_model: str | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
    ) -> None:
        # PUNTO 3 - "basandose en una variable de configuracion":
        # el argumento provider tiene prioridad; si no se entrega, se lee
        # LLM_PROVIDER desde .env; si tampoco existe, se usa "anthropic".
        self.provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
        if self.provider not in VALID_PROVIDERS:
            # Falla rapido: mejor un error claro al iniciar que uno raro despues.
            raise ValueError("LLM_PROVIDER debe ser 'openai' o 'anthropic'.")

        # Proveedor de respaldo: mismo patron (argumento > .env). Si la variable
        # no existe o esta vacia, no hay respaldo (None).
        self.fallback_provider = fallback_provider or os.getenv("LLM_FALLBACK_PROVIDER") or None
        if self.fallback_provider is not None:
            if self.fallback_provider not in VALID_PROVIDERS:
                raise ValueError("LLM_FALLBACK_PROVIDER debe ser 'openai' o 'anthropic'.")
            if self.fallback_provider == self.provider:
                raise ValueError("El proveedor de respaldo debe ser distinto del principal.")

        # Mismo patron de prioridad para los modelos: argumento > .env > default.
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.anthropic_model = anthropic_model or os.getenv(
            "ANTHROPIC_MODEL", "claude-sonnet-4-5"
        )

        # REINTENTOS Y TIMEOUT.
        # Los SDK ya reintentan solos los errores 429, 5xx y de conexion,
        # esperando cada vez un poco mas (backoff exponencial) y respetando el
        # header "retry-after" que manda el proveedor. Aca solo los hacemos
        # EXPLICITOS y configurables:
        #   max_retries -> cuantas veces reintenta el MISMO proveedor.
        #   timeout     -> segundos maximos de espera por peticion.
        # Orden real ante una caida: reintentos del principal -> respaldo.
        # Ojo: mas reintentos = mas resistente, pero el respaldo tarda mas en
        # entrar. Con max_retries=0 el respaldo entra de inmediato.
        self.max_retries = max_retries if max_retries is not None else self._env_number(
            "LLM_MAX_RETRIES", 2, int
        )
        self.timeout = timeout if timeout is not None else self._env_number(
            "LLM_TIMEOUT", 60.0, float
        )
        if self.max_retries < 0 or self.timeout <= 0:
            raise ValueError("LLM_MAX_RETRIES debe ser >= 0 y LLM_TIMEOUT > 0.")

        # PUNTO 3 - Se crean SOLO los clientes que se van a usar (principal y,
        # si hay, respaldo). Si falta una clave, el error aparece al iniciar y
        # no en medio de una caida, cuando el respaldo hace falta de verdad.
        # "Async" significa que las llamadas de red no bloquean el programa.
        self._openai: AsyncOpenAI | None = None
        self._anthropic: AsyncAnthropic | None = None
        for name in self.providers:
            if name == "openai":
                self._openai = AsyncOpenAI(
                    api_key=self._require_key("OPENAI_API_KEY"),
                    max_retries=self.max_retries,
                    timeout=self.timeout,
                )
            else:
                self._anthropic = AsyncAnthropic(
                    api_key=self._require_key("ANTHROPIC_API_KEY"),
                    max_retries=self.max_retries,
                    timeout=self.timeout,
                )

    @property
    def providers(self) -> list[str]:
        # Orden en que se intentan los proveedores: primero el principal.
        if self.fallback_provider:
            return [self.provider, self.fallback_provider]
        return [self.provider]

    @staticmethod
    def _require_key(env_var: str) -> str:
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"Falta {env_var} en el entorno.")
        return api_key

    @staticmethod
    def _env_number(env_var: str, default, cast):
        # Lee un numero desde .env. Si no esta, usa el default; si no es un
        # numero valido, da un error de configuracion claro.
        raw = os.getenv(env_var)
        if raw is None or raw.strip() == "":
            return default
        try:
            return cast(raw)
        except ValueError:
            raise ValueError(f"{env_var} debe ser un numero (recibido: {raw!r}).") from None

    # =========================================================================
    # PUNTO 4 + PUNTO 5 - Streaming con yield, control de errores y respaldo.
    # =========================================================================
    async def generate(self, params: GenerationParams) -> AsyncIterator[str | LLMError]:
        # Al tener "async def" + "yield", esta funcion es un GENERADOR ASINCRONO:
        # no devuelve toda la respuesta junta, sino que entrega fragmentos
        # uno por uno y quien la usa los recorre con "async for".
        #
        # Entrega:
        #   - str      -> fragmentos de texto del modelo (0 o mas).
        #   - LLMError -> como mucho UNO, siempre al final, si algo fallo.
        #
        # Flujo:
        #   1) Intenta con el proveedor principal (el SDK ya hizo sus reintentos).
        #   2) Si falla por una caida (RECOVERABLE_ERRORS) ANTES de enviar texto,
        #      pasa al de respaldo.
        #   3) Si ya se habia enviado texto, NO cambia: OpenAI empezaria la
        #      respuesta desde cero y el usuario veria texto duplicado.
        #   4) Errores de API -> LLMError. Bugs de nuestro codigo -> NO se
        #      esconden: se dejan ver para poder corregirlos.
        providers = self.providers
        for index, provider in enumerate(providers):
            emitted_text = False
            has_fallback = index + 1 < len(providers)
            try:
                async for chunk in self._stream_from(provider, params):
                    emitted_text = True
                    # PUNTO 4 - se reenvia cada fragmento apenas llega.
                    yield chunk
                return  # termino bien: no hace falta probar el respaldo

            # PUNTO 5 - Caidas del proveedor: rate limit, red, servidor.
            except RECOVERABLE_ERRORS as exc:
                kind = self._classify_error(exc)
                if emitted_text or not has_fallback:
                    yield LLMError(kind=kind, provider=provider, message=str(exc))
                    return
                # El aviso va a stderr (canal de errores de la terminal) para no
                # mezclarlo con el texto de la respuesta, que va a stdout.
                print(
                    f"[AVISO] {provider} no disponible ({kind}). "
                    f"Reintentando con {providers[index + 1]}...",
                    file=sys.stderr,
                )

            # PUNTO 5 - Otros errores de la API: clave invalida, peticion mal
            # armada, modelo inexistente... No se usa el respaldo porque el
            # problema es de configuracion y hay que verlo, no esconderlo.
            #
            # Antes aca habia "except Exception", que tambien atrapaba errores
            # de NUESTRO codigo (un AttributeError, un typo) y los mostraba como
            # si fueran culpa del proveedor. Ahora solo se capturan errores de
            # los SDK; un bug real se ve con su traceback completo.
            except API_ERRORS as exc:
                yield LLMError(kind="PROVIDER", provider=provider, message=str(exc))
                return

    @staticmethod
    def _classify_error(exc: Exception) -> Literal["RATE_LIMIT", "NETWORK", "SERVER"]:
        # PUNTO 5 - Traduce la clase de error a una etiqueta corta y estable.
        if isinstance(exc, RATE_LIMIT_ERRORS):
            return "RATE_LIMIT"
        if isinstance(exc, NETWORK_ERRORS):
            return "NETWORK"
        return "SERVER"

    def _stream_from(self, provider: str, params: GenerationParams) -> AsyncIterator[str]:
        # Elige la implementacion segun el nombre del proveedor.
        if provider == "openai":
            return self._generate_openai(params)
        return self._generate_anthropic(params)

    # Los metodos _generate_* NO capturan errores: los dejan subir hasta
    # generate(), que es quien decide si usar el respaldo o devolver LLMError.
    # El guion bajo inicial indica que son metodos internos.
    async def _generate_openai(self, params: GenerationParams) -> AsyncIterator[str]:
        assert self._openai is not None
        # PUNTO 4 - stream=True pide a OpenAI que devuelva eventos parciales
        # en lugar de esperar la respuesta completa.
        stream = await self._openai.chat.completions.create(
            model=self.openai_model,
            messages=[
                # Si hay system, se agrega como primer mensaje; si no, la
                # lista vacia [] hace que el * no agregue nada.
                *(
                    [{"role": "system", "content": params.system}]
                    if params.system
                    else []
                ),
                {"role": "user", "content": params.prompt},
            ],
            # Valores validados por Pydantic (PUNTO 2). Aca SI se envian
            # temperature y top_p: es lo que se usa cuando Anthropic se cae.
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            top_p=params.top_p,
            stream=True,
        )
        # Cada "event" es un pedacito de la respuesta. El texto nuevo viene
        # en choices[0].delta.content. Algunos eventos no traen texto
        # (inicio, fin, metadatos), por eso se comprueba antes de enviar.
        async for event in stream:
            if event.choices:
                content = event.choices[0].delta.content
                if content:
                    # PUNTO 4 - yield entrega el fragmento apenas llega.
                    yield content

    async def _generate_anthropic(self, params: GenerationParams) -> AsyncIterator[str]:
        assert self._anthropic is not None
        # Diferencias con OpenAI:
        #   - system va como parametro aparte, no dentro de messages.
        #   - temperature y top_p NO se envian: la version instalada del SDK
        #     (anthropic 2.5.0) ya no los acepta (ver README).
        # Se arma un diccionario para agregar system solo si existe, asi ambos
        # proveedores reciben exactamente las mismas instrucciones.
        request = {
            "model": self.anthropic_model,
            "messages": [{"role": "user", "content": params.prompt}],
            "max_tokens": params.max_tokens,
        }
        if params.system:
            request["system"] = params.system

        # PUNTO 4 - messages.stream() abre un flujo asincrono de eventos.
        # "async with" garantiza que la conexion se cierre al terminar,
        # incluso si ocurre un error en el medio.
        async with self._anthropic.messages.stream(**request) as stream:
            # text_stream ya filtra los eventos y entrega solo el texto,
            # por eso aca no hace falta revisar choices/delta como en OpenAI.
            async for text in stream.text_stream:
                # PUNTO 4 - yield entrega el fragmento apenas llega.
                yield text
