"""
test_fallback.py - Pruebas SIN llamar a las APIs (no gasta tokens ni necesita internet).

Ejecutar:
    python -m unittest -v test_fallback

Idea central: reemplazamos el metodo stream() de cada proveedor por una
funcion FALSA (fake) que devuelve texto o lanza el error que queramos. Asi
podemos "simular" una caida de Anthropic cuando queramos y comprobar que:
    PUNTO 2 -> Pydantic rechaza parametros invalidos.
    PUNTO 5 -> cada error de la API termina en un LLMError controlado.
    Extra   -> si Anthropic se cae, se usa OpenAI con la temperatura validada.
    Extra   -> se puede sumar un proveedor nuevo sin tocar llm_manager.py.
"""

import os
import unittest
from typing import AsyncIterator
from unittest.mock import patch

import anthropic
import httpx2
import openai  # libreria HTTP que usan los SDK; la necesitamos para armar errores falsos
from pydantic import ValidationError

from llm_manager import AsyncLLMManager
from providers import PROVIDERS, BaseProvider
from schemas import GenerationParams, LLMError

# Claves falsas: los clientes se crean pero nunca se conectan a nada.
FAKE_ENV = {"ANTHROPIC_API_KEY": "fake-anthropic", "OPENAI_API_KEY": "fake-openai"}

# Los errores de los SDK necesitan una peticion/respuesta HTTP para crearse.
_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def http_error(error_class, status_code: int):
    """Crea un error de API falso, ej. http_error(anthropic.RateLimitError, 429)."""
    response = httpx2.Response(status_code, request=_REQUEST)
    return error_class(f"error simulado {status_code}", response=response, body=None)


def network_error():
    return anthropic.APIConnectionError(request=_REQUEST)


def openai_network_error():
    # Cada proveedor reconoce SOLO los errores de su propio SDK, asi que una
    # caida de OpenAI se simula con un error de openai, no de anthropic.
    return openai.APIConnectionError(request=_REQUEST)


async def collect(manager, params):
    """Recorre todo el stream y separa el texto del (posible) LLMError."""
    text, errors = "", []
    async for fragment in manager.generate(params):
        if isinstance(fragment, LLMError):
            errors.append(fragment)
        else:
            text += fragment
    return text, errors


@patch.dict(os.environ, FAKE_ENV)
class FallbackTests(unittest.IsolatedAsyncioTestCase):
    # IsolatedAsyncioTestCase permite escribir tests con "async def".
    # @patch.dict pone las claves falsas solo mientras corren los tests.

    def make_manager(self, anthropic_behaviour, openai_behaviour="ok", fallback="openai"):
        """
        Crea un manager con proveedores falsos.
        *_behaviour puede ser:
            "ok"            -> responde texto normal.
            "partial"       -> manda un fragmento y despues se corta la red.
            una excepcion   -> la lanza antes de mandar texto.
        """
        manager = AsyncLLMManager(provider="anthropic", fallback_provider=fallback)
        self.openai_params = None  # aca guardamos lo que recibio OpenAI

        def fake(name, behaviour):
            async def fake_stream(params):
                if name == "openai":
                    self.openai_params = params
                if behaviour == "ok":
                    yield f"respuesta de {name}"
                    return
                if behaviour == "partial":
                    yield "Hola "
                    raise network_error()
                raise behaviour
                yield  # nunca se ejecuta; solo convierte la funcion en generador

            return fake_stream

        # Se reemplaza stream() SOLO en este objeto (no en la clase), asi el
        # resto del proveedor (classify_error, nombre...) sigue siendo el real.
        manager.get_provider("anthropic").stream = fake("anthropic", anthropic_behaviour)
        if fallback:
            manager.get_provider("openai").stream = fake("openai", openai_behaviour)
        return manager

    def params(self):
        return GenerationParams(prompt="hola", temperature=1.3, top_p=0.9)

    # --- Caso feliz -----------------------------------------------------------

    async def test_anthropic_ok_no_usa_respaldo(self):
        text, errors = await collect(self.make_manager("ok"), self.params())
        self.assertEqual(text, "respuesta de anthropic")
        self.assertEqual(errors, [])
        self.assertIsNone(self.openai_params, "OpenAI no deberia haberse llamado")

    # --- Caidas que activan el respaldo (PUNTO 5 + extra) -------------------

    async def test_caidas_usan_openai_con_la_temperatura_validada(self):
        caidas = {
            "red": network_error(),
            "rate limit 429": http_error(anthropic.RateLimitError, 429),
            "servidor 500": http_error(anthropic.InternalServerError, 500),
            "saturado 529": http_error(anthropic.OverloadedError, 529),
        }
        for nombre, error in caidas.items():
            # subTest: si falla un caso, el reporte dice cual.
            with self.subTest(caida=nombre):
                text, errors = await collect(self.make_manager(error), self.params())
                self.assertEqual(text, "respuesta de openai")
                self.assertEqual(errors, [])
                self.assertEqual(self.openai_params.temperature, 1.3)
                self.assertEqual(self.openai_params.top_p, 0.9)

    # --- Casos que NO deben usar el respaldo ----------------------------------

    async def test_clave_invalida_no_usa_respaldo(self):
        error = http_error(anthropic.AuthenticationError, 401)
        text, errors = await collect(self.make_manager(error), self.params())
        self.assertEqual(text, "")
        self.assertEqual([e.kind for e in errors], ["PROVIDER"])
        self.assertIsNone(self.openai_params)

    async def test_caida_a_mitad_no_duplica_texto(self):
        text, errors = await collect(self.make_manager("partial"), self.params())
        self.assertEqual(text, "Hola ")
        self.assertEqual([e.kind for e in errors], ["NETWORK"])
        self.assertIsNone(self.openai_params)

    async def test_sin_respaldo_configurado_devuelve_error(self):
        # fallback=None significa "leer .env", asi que tambien vaciamos la variable.
        with patch.dict(os.environ, {"LLM_FALLBACK_PROVIDER": ""}):
            manager = self.make_manager(http_error(anthropic.RateLimitError, 429), fallback=None)
        text, errors = await collect(manager, self.params())
        self.assertEqual([(e.kind, e.provider) for e in errors], [("RATE_LIMIT", "anthropic")])

    async def test_si_el_respaldo_tambien_falla_devuelve_error(self):
        manager = self.make_manager(network_error(), openai_behaviour=openai_network_error())
        text, errors = await collect(manager, self.params())
        self.assertEqual(text, "")
        self.assertEqual([(e.kind, e.provider) for e in errors], [("NETWORK", "openai")])

    # --- Bugs propios: NO se esconden -----------------------------------------

    async def test_un_bug_de_codigo_no_se_disfraza_de_error_de_api(self):
        manager = self.make_manager(AttributeError("bug en nuestro codigo"))
        with self.assertRaises(AttributeError):
            await collect(manager, self.params())


# --- Extensibilidad: un tercer proveedor sin tocar llm_manager.py -------------

class EchoProvider(BaseProvider):
    """Proveedor de juguete: repite el prompt. No usa red ni ningun SDK."""

    name = "echo"
    api_key_env = "ECHO_API_KEY"
    model_env = "ECHO_MODEL"
    default_model = "echo-1"
    api_error = ConnectionRefusedError  # cualquier clase de error sirve de ejemplo

    def create_client(self, api_key, max_retries, timeout):
        return None  # no necesita cliente

    async def stream(self, params: GenerationParams) -> AsyncIterator[str]:
        for word in params.prompt.split():
            yield word + " "


@patch.dict(os.environ, {**FAKE_ENV, "ECHO_API_KEY": "fake-echo"})
@patch.dict(PROVIDERS, {"echo": EchoProvider})
class ExtensibilityTests(unittest.IsolatedAsyncioTestCase):
    # @patch.dict(PROVIDERS, ...) registra "echo" solo durante este test.
    # Es exactamente lo que haria un proveedor real: agregar UNA linea a
    # PROVIDERS. AsyncLLMManager no se modifico para nada.

    async def test_nuevo_proveedor_funciona_sin_cambiar_el_manager(self):
        manager = AsyncLLMManager(provider="echo", fallback_provider="")
        text, errors = await collect(manager, GenerationParams(prompt="hola mundo"))
        self.assertEqual(text, "hola mundo ")
        self.assertEqual(errors, [])

    async def test_nuevo_proveedor_puede_ser_respaldo(self):
        manager = AsyncLLMManager(provider="anthropic", fallback_provider="echo")

        async def caida(params):
            raise network_error()
            yield

        manager.get_provider("anthropic").stream = caida
        text, errors = await collect(manager, GenerationParams(prompt="respaldo echo"))
        self.assertEqual(text, "respaldo echo ")


@patch.dict(os.environ, FAKE_ENV)
class ConfigTests(unittest.TestCase):

    def test_reintentos_y_timeout_llegan_a_los_clientes(self):
        manager = AsyncLLMManager(
            provider="anthropic", fallback_provider="openai", max_retries=0, timeout=15
        )
        self.assertEqual(manager.get_provider("anthropic").client.max_retries, 0)
        self.assertEqual(manager.get_provider("openai").client.timeout, 15)

    def test_modelo_por_argumento(self):
        manager = AsyncLLMManager(
            provider="anthropic", fallback_provider="openai", models={"openai": "gpt-test"}
        )
        self.assertEqual(manager.get_provider("openai").model, "gpt-test")

    def test_proveedor_desconocido_es_invalido(self):
        with self.assertRaises(ValueError):
            AsyncLLMManager(provider="inexistente")

    def test_respaldo_vacio_desactiva_el_de_env(self):
        # Bug encontrado probando con las APIs reales: fallback_provider=""
        # se ignoraba y se usaba el respaldo del .env igual.
        with patch.dict(os.environ, {"LLM_FALLBACK_PROVIDER": "openai"}):
            manager = AsyncLLMManager(provider="openai", fallback_provider="")
            self.assertEqual(manager.providers, ["openai"])
            # None (no indicarlo) si sigue leyendo el .env:
            manager = AsyncLLMManager(provider="anthropic")
            self.assertEqual(manager.providers, ["anthropic", "openai"])

    def test_respaldo_igual_al_principal_es_invalido(self):
        with self.assertRaises(ValueError):
            AsyncLLMManager(provider="anthropic", fallback_provider="anthropic")

    def test_falta_clave_del_respaldo(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaises(ValueError):
                AsyncLLMManager(provider="anthropic", fallback_provider="openai")


class ParamsTests(unittest.TestCase):
    # PUNTO 2 - El esquema Pydantic (schemas.py) rechaza valores fuera de rango.

    def test_rechaza_parametros_invalidos(self):
        invalidos = [
            {"prompt": "   "},
            {"prompt": "hola", "temperature": 2.5},
            {"prompt": "hola", "temperature": -0.1},
            {"prompt": "hola", "max_tokens": 0},
            {"prompt": "hola", "top_p": 0},
        ]
        for datos in invalidos:
            with self.subTest(datos=datos):
                with self.assertRaises(ValidationError):
                    GenerationParams(**datos)

    def test_acepta_los_limites(self):
        GenerationParams(prompt="hola", temperature=0)
        GenerationParams(prompt="hola", temperature=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
