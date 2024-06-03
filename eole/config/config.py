from pydantic import BaseModel, ConfigDict  # , Enum


class Config(BaseModel):
    """
    All config classes will inherit from this.
    Allows to homogenize model_config settings across all classes,
    and clarifies a bit naming standards (pydantic model_config vs our own).
    """

    model_config = ConfigDict(
        validate_assignment=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
    )
    model_config[
        "protected_namespaces"
    ] = ()  # prevents warning for model_task / model_type
