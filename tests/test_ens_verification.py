from typing import Optional

from agent0_sdk.core.agent import Agent
from agent0_sdk.core.ens_verification import (
    AgentRegistryRecord,
    build_agent_registry_record_key,
    decode_agent_registry_value,
    load_agent_registry_record,
    record_matches_agent,
)
from agent0_sdk.core.models import Endpoint, EndpointType, RegistrationFile


class DummyResolver:
    def __init__(self, value: str):
        self._value = value
        self.last_key = None

    def get_text(self, key: str) -> str:
        self.last_key = key
        return self._value


class DummyProvider:
    def __init__(self, resolver: Optional[DummyResolver]):
        self._resolver = resolver
        self.ens = self.DummyENS(resolver)

    class DummyENS:
        def __init__(self, resolver: Optional[DummyResolver]):
            self._resolver = resolver

        def get_text(self, ens_name: str, key: str):
            if not self._resolver:
                raise ValueError(f"No resolver for {ens_name}")
            return self._resolver.get_text(key)


class DummyWeb3Client:
    def __init__(self, provider):
        self.w3 = provider


class DummyIdentityRegistry:
    def __init__(self, address: str):
        self.address = address


class DummySDK:
    def __init__(self, chain_id: int, provider, registry_address: str):
        self.chainId = chain_id
        self.web3_client = DummyWeb3Client(provider)
        self.identity_registry = DummyIdentityRegistry(registry_address)


def _agent_registry_value(registry_address: str, agent_id: int) -> str:
    if agent_id < 0:
        raise ValueError("agent_id must be non-negative")

    if agent_id == 0:
        agent_bytes = b""
    else:
        agent_bytes = agent_id.to_bytes((agent_id.bit_length() + 7) // 8, "big")

    return f"{registry_address}{len(agent_bytes):02x}{agent_bytes.hex()}"


def _make_agent(sdk: DummySDK, agent_id: str, ens_name: str) -> Agent:
    registration = RegistrationFile(
        agentId=agent_id,
        endpoints=[Endpoint(type=EndpointType.ENS, value=ens_name, meta={})],
    )
    return Agent(sdk=sdk, registration_file=registration)


class TestBuildAgentRegistryRecordKey:
    def test_encodes_chain_id(self):
        key = build_agent_registry_record_key(8453)
        assert key == "agent-registry:0001000002210500"


class TestDecodeAgentRegistryValue:
    def test_single_byte_agent_id(self):
        address = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
        value = f"{address}01a7"
        record = decode_agent_registry_value(value)
        assert record.address == address
        assert record.agent_id == 0xA7
        assert record.chain_reference == 0


class TestVerifyENSName:
    def test_success(self):
        chain_id = 8453
        token_id = 1234
        ens_name = "agent.example.eth"
        registry_address = "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

        value = _agent_registry_value(registry_address, token_id)
        resolver = DummyResolver(value)
        provider = DummyProvider(resolver)
        sdk = DummySDK(chain_id, provider, registry_address)
        agent = _make_agent(sdk, f"{chain_id}:{token_id}", ens_name)

        assert agent.verifyENSName() is True
        expected_key = build_agent_registry_record_key(chain_id)
        assert resolver.last_key == expected_key

    def test_wrong_agent_id(self):
        chain_id = 8453
        registry_address = "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

        value = _agent_registry_value(registry_address, 1234)
        resolver = DummyResolver(value)
        provider = DummyProvider(resolver)
        sdk = DummySDK(chain_id, provider, registry_address)
        agent = _make_agent(sdk, f"{chain_id}:999", "example.eth")

        assert agent.verifyENSName() is False

    def test_wrong_registry(self):
        chain_id = 8453
        registry_address = "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
        other_address = "0x0000000000000000000000000000000000000001"

        value = _agent_registry_value(registry_address, 1234)
        resolver = DummyResolver(value)
        provider = DummyProvider(resolver)
        sdk = DummySDK(chain_id, provider, other_address)
        agent = _make_agent(sdk, f"{chain_id}:1234", "example.eth")

        assert agent.verifyENSName() is False

    def test_missing_resolver(self):
        chain_id = 8453
        registry_address = "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

        provider = DummyProvider(resolver=None)
        sdk = DummySDK(chain_id, provider, registry_address)
        agent = _make_agent(sdk, f"{chain_id}:1234", "example.eth")

        assert agent.verifyENSName() is False


class TestRecordMatchesAgent:
    def test_validates_fields(self):
        from eth_utils import to_checksum_address

        record = AgentRegistryRecord(
            version=1,
            chain_type=0,
            chain_reference=10,
            address=to_checksum_address("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"),
            agent_id=5,
        )
        assert record_matches_agent(
            record,
            {
                "chainId": 10,
                "registryAddress": "0xa0B86991c6218B36c1d19D4A2e9Eb0ce3606Eb48",
                "agentId": 5,
            },
        )
