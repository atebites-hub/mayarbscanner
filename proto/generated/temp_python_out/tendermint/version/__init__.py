# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: tendermint/version/types.proto
# plugin: python-betterproto
# This file has been @generated

from dataclasses import dataclass

import betterproto


@dataclass(eq=False, repr=False)
class App(betterproto.Message):
    """
    App includes the protocol and software version for the application.
     This information is included in ResponseInfo. The App.Protocol can be
     updated in ResponseEndBlock.
    """

    protocol: int = betterproto.uint64_field(1)
    software: str = betterproto.string_field(2)


@dataclass(eq=False, repr=False)
class Consensus(betterproto.Message):
    """
    Consensus captures the consensus rules for processing a block in the blockchain,
     including all blockchain data structures and the rules of the application's
     state transition machine.
    """

    block: int = betterproto.uint64_field(1)
    app: int = betterproto.uint64_field(2)
