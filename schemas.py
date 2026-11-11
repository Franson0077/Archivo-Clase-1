"""
schemas.py - Esquemas de datos del proyecto (modelos Pydantic).

Aca viven SOLO las estructuras de datos: que entra al modelo y que sale
cuando algo falla. No hay logica de red ni de proveedores; por eso este
archivo no importa openai ni anthropic.

Mapa de la consigna dentro de este archivo:
    PUNTO 2 -> GenerationParams (validacion de parametros de entrada).
    PUNTO 5 -> LLMError (error controlado devuelto como objeto).
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# Tipos de error posibles. Se define una sola vez y se reutiliza en
# LLMError y en providers.py (classify_error), asi no hay que repetirlo.
ErrorKind = Literal["RATE_LIMIT", "NETWORK", "SERVER", "PROVIDER"]


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
    # generate() entrega textos (str) o, si algo falla, UN LLMError.
    # Quien consume pregunta: isinstance(fragmento, LLMError), sin tener que
    # leer el texto para adivinar si es un error.
    #
    # Para un AGENTE esto es clave: puede decidir que hacer segun el tipo de
    # error (esperar, avisar, cambiar de herramienta...) en vez de solo mostrarlo.
    #
    # Alcance: LLMError cubre los errores que vienen de la API. Los errores de
    # configuracion (clave faltante) y los bugs de nuestro codigo NO se
    # convierten en LLMError (ver README, "Que errores se controlan").
    kind: ErrorKind
    provider: str
    message: str

    def __str__(self) -> str:
        # Formato legible para mostrar en la terminal.
        return f"[ERROR:{self.kind}] {self.provider}: {self.message}"
