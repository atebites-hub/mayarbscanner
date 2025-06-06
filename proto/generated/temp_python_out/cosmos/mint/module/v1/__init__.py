# Generated by the protocol buffer compiler.  DO NOT EDIT!
# sources: cosmos/mint/module/v1/module.proto
# plugin: python-betterproto
# This file has been @generated

from dataclasses import dataclass

import betterproto


@dataclass(eq=False, repr=False)
class Module(betterproto.Message):
    """Module is the config object of the mint module."""

    fee_collector_name: str = betterproto.string_field(1)
    authority: str = betterproto.string_field(2)
    """
    authority defines the custom module authority. If not set, defaults to the governance module.
    """
