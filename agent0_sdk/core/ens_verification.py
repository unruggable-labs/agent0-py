"""
Utilities used by `Agent.verifyENSName()` to read and validate ENSIP-25 records.

We fetch the ENS text key `agent-registry:<7930 chain id>`,
decode the value (`<EIP55 registry address><agentIdLength><agentIdHex>`), then
compare the decoded registry + token id against what the SDK expects.
ENSIP-25:
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

from eth_utils import to_checksum_address

# Currently restricted to EVM (eip155) namespaces; future namespaces can be added here.
CAIP_NAMESPACE_CODES = {"eip155": 0x0000}
CAIP_VERSION = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentRegistryRecord:
    """Decoded representation of the ENS registry record."""

    version: int
    chain_type: int
    chain_reference: int
    address: str
    agent_id: int


def _normalize_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 0)
    raise TypeError(f"Unsupported numeric type: {type(value)!r}")


def _chain_reference_bytes(chain_id: int) -> bytes:
    if chain_id < 0:
        raise ValueError("Chain reference must be non-negative")
    if chain_id == 0:
        return b"\x00"
    length = (chain_id.bit_length() + 7) // 8
    return chain_id.to_bytes(length, "big")


def _encode_erc7930_chain_identifier(chain_id: int, namespace: str) -> str:
    """Encode the ERC-7930 chain identifier (no address), as lowercase hex without leading 0x."""
    if namespace not in CAIP_NAMESPACE_CODES:
        raise ValueError(f"Unsupported CAIP namespace: {namespace}")

    version_bytes = CAIP_VERSION.to_bytes(2, "big")
    chain_type_bytes = CAIP_NAMESPACE_CODES[namespace].to_bytes(2, "big")
    chain_reference_bytes = _chain_reference_bytes(chain_id)
    chain_reference_length = len(chain_reference_bytes).to_bytes(1, "big")
    # addrLen=0 per ERC-7930 because the chain identifier envelope does not include an address.
    address_length = (0).to_bytes(1, "big")

    payload = (
        version_bytes
        + chain_type_bytes
        + chain_reference_length
        + chain_reference_bytes
        + address_length
    )
    return payload.hex()


def build_agent_registry_record_key(
    chain_id: Any, namespace: str = "eip155"
) -> str:
    """
    Build key: `agent-registry:<hex 7930 chain id>` (EVM only).
    See ERC-7930 for chain id binary envelope (addrLen=0), hex-encoded.
    """
    normalized_chain_id = _normalize_int(chain_id)
    envelope_hex = _encode_erc7930_chain_identifier(normalized_chain_id, namespace)
    return f"agent-registry:{envelope_hex}"


def decode_agent_registry_value(value: str) -> AgentRegistryRecord:
    """
    Decode value text: `<EIP55 address><agentIdLen><agentId>`.
    Validates minimum length and hex sizes; normalizes address via EIP-55.
    """
    if not isinstance(value, str):
        raise TypeError("Agent registry value must be a string")

    if not value.startswith("0x"):
        raise ValueError("Agent registry record must start with 0x")

    address_section_length = 42  # 0x + 40 hex chars for EVM addresses
    # Require checksum address (42 chars) + at least one byte for agentId length.
    if len(value) < address_section_length + 2:
        raise ValueError("Agent registry record value too short")

    # Split the payload into `<address><len><agentId>` per ENSIP-25.
    address_text = value[:address_section_length]
    length_hex = value[address_section_length : address_section_length + 2]
    try:
        agent_id_length = int(length_hex, 16)
    except ValueError as exc:
        raise ValueError("Invalid agent ID length") from exc

    if agent_id_length < 0:
        raise ValueError("Agent ID length must be non-negative")

    agent_hex = value[address_section_length + 2 :].lower()
    try:
        agent_bytes = bytes.fromhex(agent_hex)
    except ValueError as exc:
        raise ValueError("Agent ID must be valid hex") from exc

    if len(agent_bytes) != agent_id_length:
        raise ValueError("Agent ID length does not match payload")

    normalized_address = to_checksum_address(address_text)
    agent_id = int.from_bytes(agent_bytes, "big") if agent_bytes else 0

    return AgentRegistryRecord(
        version=1,
        chain_type=CAIP_NAMESPACE_CODES["eip155"],
        chain_reference=0,
        address=normalized_address,
        agent_id=agent_id,
    )


def load_agent_registry_record(
    provider: Any,
    ens_name: str,
    chain_id: Any,
    namespace: str = "eip155",
) -> Optional[AgentRegistryRecord]:
    """
    Load and decode an ENS agent-registry record for a specific chain.

    @param provider The SDK's provider.
    @param ensName ENS name to inspect (e.g., `agent.eth`).
    @param chainId Chain identifier encoded in the ENSIP key (eip155 only).
    @param namespace CAIP namespace (defaults to `eip155`; other namespaces not yet supported).
    """
    # Build the ENS text key that points to the agent registry payload.
    record_key = build_agent_registry_record_key(chain_id, namespace)

    ens_api = getattr(provider, "ens", None)
    if ens_api is None:
        logger.debug("Provider has no ENS support; cannot resolve %s", ens_name)
        return None

    try:
        value = ens_api.get_text(ens_name, record_key)
    except Exception as exc:
        logger.debug(
            "Failed to fetch ENS text record %s for %s: %s",
            record_key,
            ens_name,
            exc,
        )
        return None

    if not value:
        logger.debug("ENS text record %s missing for %s", record_key, ens_name)
        return None

    try:
        # Decode the ENS record.
        decoded = decode_agent_registry_value(value)
    except Exception as exc:
        logger.debug(
            "Failed to decode ENS text record %s for %s: %s",
            record_key,
            ens_name,
            exc,
        )
        return None

    return replace(decoded, chain_reference=_normalize_int(chain_id))


def record_matches_agent(
    record: AgentRegistryRecord,
    expected: Dict[str, Any],
) -> bool:
    """Compare a decoded record against expected agent data (EVM only)."""
    # Currently limited to EVM-compatible records (eip155 namespace).
    if record.version != 1 or record.chain_type != CAIP_NAMESPACE_CODES["eip155"]:
        return False

    try:
        expected_chain_id = _normalize_int(expected.get("chainId"))
        expected_agent_id = _normalize_int(expected.get("agentId"))
    except Exception:
        return False

    # The ENS record must be anchored to the same chain, registry contract, and token id.
    if record.chain_reference != expected_chain_id:
        return False

    try:
        expected_address = to_checksum_address(expected.get("registryAddress"))
    except Exception:
        return False

    if record.address != expected_address:
        return False

    if record.agent_id != expected_agent_id:
        return False

    return True
