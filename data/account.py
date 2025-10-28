from dataclasses import dataclass


@dataclass
class Account:
    name: str
    balance: float
    exp_return: float
    vol: float = 0.0
    pre_tax_contrib_rate: float = 0.0   # % of gross salary before tax
    post_tax_contrib_rate: float = 0.0  # % of net salary after tax
    is_income_sink: bool = False        # salary/bonus land here
    is_expense_source: bool = False     # expenses deducted from here first
    is_isa: bool = False                # ISA flag, contributions capped globally
