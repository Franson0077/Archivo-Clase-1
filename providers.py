"""
providers.py - Un proveedor de LLM por clase, todos con la misma "forma".

Idea central (patron Strategy):
    BaseProvider define QUE tiene que saber hacer cualquier proveedor:
        - crear su cliente (create_client),
        - generar texto en streaming (stream),
        - reconocer sus propios errores (classify_error).
    Cada subclase (OpenAIProvider, AnthropicProvider) define COMO lo hace.

    AsyncLLMManager solo trabaja con BaseProvider: no sabe si del otro lado
    hay OpenAI, Anthropic u otro. Por eso no tiene ramas if/else por proveedor.

Como agregar un tercer proveedor (ej. "mistral"):
    1) Crear class MistralProvider(BaseProvider) con sus atributos y metodos.
    2) Agregarlo al diccionario PROVIDERS al final de este archivo.
    No hace falta tocar llm_manager.py ni main.py.

Mapa de la consigna dentro de este archivo:
    PUNTO 1 -> imports de openai y anthropic.
    PUNTO 3 -> PROVIDERS: los proveedores que se pueden cargar por configuracion.
    PUNTO 4 -> metodo stream() de cada proveedor (streaming con yield).
    PUNTO 5 -> classify_error(): reconoce rate limit, red y servidor.
"""

import os

# ABC = "Abstract Base Class". Una clase abstracta no se puede usar directo:
# sirve de molde. @abstractmethod marca los metodos que cada subclase esta
# OBLIGADA a implementar; si se olvida uno, Python da error al crearla.
from abc import ABC, abstractmethod

# ClassVar: atributo que pertenece a la CLASE (igual para todas sus instancias),
# por ejemplo el nombre "openai" o la variable de entorno de su clave.
from typing import Any, AsyncIterator, ClassVar

# PUNTO 1 - Dependencias pedidas por la consigna (openai y anthropic).
# Se renombran con "as" porque ambos SDK usan los mismos nombres de clase.
import anthropic
import openai

from schemas import ErrorKind, GenerationParams


class BaseProvider(ABC):
    # --- Lo que cada subclase DEBE completar ---------------------------------
    name: ClassVar[str]           # nombre usado en .env, ej. "openai"
    api_key_env: ClassVar[str]    # variable con la clave, ej. "OPENAI_API_KEY"
    model_env: ClassVar[str]      # variable con el modelo, ej. "OPENAI_MODEL"
    default_model: ClassVar[str]  # modelo si la variable no existe

    # PUNTO 5 - Clases de error de ESTE proveedor, agrupadas por categoria.
    # Son tuplas porque isinstance(exc, (A, B, C)) acepta tuplas.
    rate_limit_errors: ClassVar[tuple[type[Exception], ...]] = ()
    network_errors: ClassVar[tuple[type[Exception], ...]] = ()
    server_errors: ClassVar[tuple[type[Exception], ...]] = ()
    # Clase madre de TODOS los errores del SDK (clave invalida, peticion mal
    # armada...). Lo que no este en las tuplas de arriba cae aca como PROVIDER.
    api_error: ClassVar[type[Exception]]

    def __init__(self, model: str | None, max_retries: int, timeout: float) -> None:
        # Mismo patron de prioridad para el modelo: argumento > .env > default.
        self.model = model or os.getenv(self.model_env, self.default_model)

        # Si falta la clave, se avisa al CREAR el proveedor (al iniciar el
        # programa) y no en medio de una caida, cuando el respaldo hace falta.
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"Falta {self.api_key_env} en el entorno.")

        # REINTENTOS Y TIMEOUT: los SDK reintentan solos los errores 429, 5xx
        # y de conexion (esperando cada vez un poco mas). Aca solo los hacemos
        # explicitos. "Async" = las llamadas de red no bloquean el programa.
        self.client = self.create_client(api_key, max_retries, timeout)

    @abstractmethod
    def create_client(self, api_key: str, max_retries: int, timeout: float) -> Any:
        """Crea el cliente asincrono del SDK."""

    @abstractmethod
    def stream(self, params: GenerationParams) -> AsyncIterator[str]:
        """
        PUNTO 4 - Genera texto en streaming: entrega fragmentos con yield.
        NO captura errores: los deja subir hasta el manager, que decide si
        usar el respaldo o devolver un LLMError.
        """

    def classify_error(self, exc: Exception) -> ErrorKind | None:
        # PUNTO 5 - Traduce un error de ESTE SDK a una etiqueta corta.
        # Devuelve None si NO es un error de la API (ej. un bug de nuestro
        # codigo): el manager lo deja subir en vez de disfrazarlo.
        if isinstance(exc, self.rate_limit_errors):
            return "RATE_LIMIT"
        if isinstance(exc, self.network_errors):
            return "NETWORK"
        if isinstance(exc, self.server_errors):
            return "SERVER"
        if isinstance(exc, self.api_error):
            return "PROVIDER"
        return None


# =============================================================================
# OpenAI
# =============================================================================
class OpenAIProvider(BaseProvider):
    name = "openai"
    api_key_env = "OPENAI_API_KEY"
    model_env = "OPENAI_MODEL"
    default_model = "gpt-4o-mini"

    rate_limit_errors = (openai.RateLimitError,)
    network_errors = (openai.APIConnectionError, openai.APITimeoutError)
    server_errors = (openai.InternalServerError,)
    api_error = openai.APIError

    def create_client(self, api_key: str, max_retries: int, timeout: float) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(api_key=api_key, max_retries=max_retries, timeout=timeout)

    async def stream(self, params: GenerationParams) -> AsyncIterator[str]:
        # PUNTO 4 - stream=True pide a OpenAI que devuelva eventos parciales
        # en lugar de esperar la respuesta completa.
        response = await self.client.chat.completions.create(
            model=self.model,
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
        async for event in response:
            if event.choices:
                content = event.choices[0].delta.content
                if content:
                    # PUNTO 4 - yield entrega el fragmento apenas llega.
                    yield content


# =============================================================================
# Anthropic
# =============================================================================
class AnthropicProvider(BaseProvider):
    name = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    model_env = "ANTHROPIC_MODEL"
    default_model = "claude-sonnet-4-5"

    rate_limit_errors = (anthropic.RateLimitError,)
    network_errors = (anthropic.APIConnectionError, anthropic.APITimeoutError)
    # InternalServerError incluye OverloadedError (529, "servidor saturado").
    server_errors = (anthropic.InternalServerError,)
    api_error = anthropic.APIError

    def create_client(self, api_key: str, max_retries: int, timeout: float) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(api_key=api_key, max_retries=max_retries, timeout=timeout)

    async def stream(self, params: GenerationParams) -> AsyncIterator[str]:
        # Diferencias con OpenAI:
        #   - system va como parametro aparte, no dentro de messages.
        #   - temperature y top_p NO se envian: la version instalada del SDK
        #     (anthropic 2.5.0) ya no los acepta (ver README).
        # Se arma un diccionario para agregar system solo si existe, asi ambos
        # proveedores reciben exactamente las mismas instrucciones.
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": params.prompt}],
            "max_tokens": params.max_tokens,
        }
        if params.system:
            request["system"] = params.system

        # PUNTO 4 - messages.stream() abre un flujo asincrono de eventos.
        # "async with" garantiza que la conexion se cierre al terminar,
        # incluso si ocurre un error en el medio.
        async with self.client.messages.stream(**request) as response:
            # text_stream ya filtra los eventos y entrega solo el texto.
            async for text in response.text_stream:
                # PUNTO 4 - yield entrega el fragmento apenas llega.
                yield text


# =============================================================================
# PUNTO 3 - Registro de proveedores disponibles.
# =============================================================================
# El valor de LLM_PROVIDER / LLM_FALLBACK_PROVIDER se busca en este diccionario.
# Para sumar un proveedor nuevo, se agrega UNA linea aca.
PROVIDERS: dict[str, type[BaseProvider]] = {
    OpenAIProvider.name: OpenAIProvider,
    AnthropicProvider.name: AnthropicProvider,
}
