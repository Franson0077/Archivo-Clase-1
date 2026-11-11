"""
llm_manager.py - Orquestador: elige el proveedor, maneja el respaldo y los errores.

Este archivo NO sabe como hablar con OpenAI o Anthropic: eso esta en
providers.py. Tampoco define los datos: eso esta en schemas.py.
Aca solo se decide QUE proveedor usar y QUE hacer si falla.

Mapa de la consigna dentro de este archivo:
    PUNTO 1 -> import de python-dotenv (load_dotenv).
    PUNTO 3 -> AsyncLLMManager.__init__ (carga el proveedor segun LLM_PROVIDER).
    PUNTO 4 -> AsyncLLMManager.generate (streaming con yield).
    PUNTO 5 -> AsyncLLMManager.generate (errores de la API -> LLMError).

Extras (no pedidos por la consigna):
    - Proveedor de RESPALDO: si el principal (ej. Anthropic) se cae,
      generate() reintenta con LLM_FALLBACK_PROVIDER (ej. OpenAI), usando la
      temperatura y top_p validados por Pydantic.
    - Reintentos y timeout configurables (LLM_MAX_RETRIES, LLM_TIMEOUT).
"""

import os
import sys
from typing import AsyncIterator

# PUNTO 1 - Dependencia pedida por la consigna (python-dotenv).
# load_dotenv() busca un archivo .env y carga sus pares clave=valor en os.environ.
# Asi las claves quedan fuera del codigo y pueden cambiar sin modificarlo.
from dotenv import load_dotenv

from providers import PROVIDERS, BaseProvider
from schemas import GenerationParams, LLMError

# Se re-exportan para que "from llm_manager import GenerationParams, LLMError"
# siga funcionando (asi lo usan main.py y test_fallback.py).
__all__ = ["AsyncLLMManager", "GenerationParams", "LLMError"]


# Se ejecuta una sola vez, al importar el modulo. Desde aca en adelante
# os.getenv("...") ya puede leer las variables definidas en .env.
load_dotenv()


# =============================================================================
# PUNTO 3 - Clase AsyncLLMManager: carga el proveedor segun configuracion.
# =============================================================================
class AsyncLLMManager:
    # Prepara un proveedor principal y, opcionalmente, uno de respaldo.
    # El resto del programa (main.py) no necesita saber que proveedor se usa:
    # siempre llama a manager.generate(params) y recibe texto o un LLMError.
    def __init__(
        self,
        provider: str | None = None,
        fallback_provider: str | None = None,
        models: dict[str, str] | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
    ) -> None:
        # PUNTO 3 - "basandose en una variable de configuracion":
        # el argumento provider tiene prioridad; si no se entrega, se lee
        # LLM_PROVIDER desde .env; si tampoco existe, se usa "anthropic".
        self.provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
        self._check_known(self.provider, "LLM_PROVIDER")

        # Proveedor de respaldo: mismo patron (argumento > .env). Si la variable
        # no existe o esta vacia, no hay respaldo (None).
        self.fallback_provider = fallback_provider or os.getenv("LLM_FALLBACK_PROVIDER") or None
        if self.fallback_provider is not None:
            self._check_known(self.fallback_provider, "LLM_FALLBACK_PROVIDER")
            if self.fallback_provider == self.provider:
                raise ValueError("El proveedor de respaldo debe ser distinto del principal.")

        # Reintentos y timeout (se le pasan a cada proveedor).
        #   max_retries -> cuantas veces reintenta el MISMO proveedor.
        #   timeout     -> segundos maximos de espera por peticion.
        # Orden real ante una caida: reintentos del principal -> respaldo.
        # Con max_retries=0 el respaldo entra de inmediato.
        self.max_retries = max_retries if max_retries is not None else _env_number(
            "LLM_MAX_RETRIES", 2, int
        )
        self.timeout = timeout if timeout is not None else _env_number(
            "LLM_TIMEOUT", 60.0, float
        )
        if self.max_retries < 0 or self.timeout <= 0:
            raise ValueError("LLM_MAX_RETRIES debe ser >= 0 y LLM_TIMEOUT > 0.")

        # PUNTO 3 - Se crean SOLO los proveedores que se van a usar, en orden
        # (principal primero). No hay if/else por proveedor: se busca la clase
        # en PROVIDERS y se crea. "models" permite elegir el modelo de cada uno
        # por codigo, ej. models={"openai": "gpt-4o"}; si no, se lee de .env.
        models = models or {}
        self._chain: list[BaseProvider] = [
            PROVIDERS[name](
                model=models.get(name),
                max_retries=self.max_retries,
                timeout=self.timeout,
            )
            for name in self.providers
        ]

    @property
    def providers(self) -> list[str]:
        # Nombres de los proveedores en el orden en que se intentan.
        if self.fallback_provider:
            return [self.provider, self.fallback_provider]
        return [self.provider]

    def get_provider(self, name: str) -> BaseProvider:
        # Devuelve el objeto proveedor ya creado (util en tests y para depurar).
        for provider in self._chain:
            if provider.name == name:
                return provider
        raise KeyError(f"El proveedor {name!r} no esta cargado en este manager.")

    @staticmethod
    def _check_known(name: str, env_var: str) -> None:
        # Falla rapido: mejor un error claro al iniciar que uno raro despues.
        # La lista de opciones sale de PROVIDERS, asi se actualiza sola.
        if name not in PROVIDERS:
            options = ", ".join(repr(n) for n in PROVIDERS)
            raise ValueError(f"{env_var} debe ser uno de: {options} (recibido: {name!r}).")

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
        #   2) Si falla por una caida (RATE_LIMIT, NETWORK, SERVER) ANTES de
        #      enviar texto, pasa al de respaldo.
        #   3) Si ya se habia enviado texto, NO cambia: el respaldo empezaria la
        #      respuesta desde cero y el usuario veria texto duplicado.
        #   4) Error de configuracion de la API (PROVIDER: clave invalida...)
        #      -> LLMError sin respaldo: hay que verlo y corregirlo.
        #   5) Error que no viene de la API (bug de nuestro codigo) -> NO se
        #      esconde: se deja subir con su traceback completo.
        for index, provider in enumerate(self._chain):
            emitted_text = False
            has_fallback = index + 1 < len(self._chain)
            try:
                async for chunk in provider.stream(params):
                    emitted_text = True
                    # PUNTO 4 - se reenvia cada fragmento apenas llega.
                    yield chunk
                return  # termino bien: no hace falta probar el respaldo

            except Exception as exc:
                # PUNTO 5 - Cada proveedor reconoce sus propios errores.
                kind = provider.classify_error(exc)
                if kind is None:
                    raise  # no es un error de la API: es un bug, que se vea

                is_outage = kind in ("RATE_LIMIT", "NETWORK", "SERVER")
                if is_outage and has_fallback and not emitted_text:
                    # El aviso va a stderr (canal de errores de la terminal)
                    # para no mezclarlo con el texto de la respuesta (stdout).
                    print(
                        f"[AVISO] {provider.name} no disponible ({kind}). "
                        f"Reintentando con {self._chain[index + 1].name}...",
                        file=sys.stderr,
                    )
                    continue

                yield LLMError(kind=kind, provider=provider.name, message=str(exc))
                return


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
