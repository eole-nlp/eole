from pydantic import BaseModel, ConfigDict  # , Enum


def get_config_dict():
    return ConfigDict(
        validate_assignment=True,
        validate_default=True,
        use_enum_values=True,
        extra="forbid",
        protected_namespaces=(),  # prevents warning for model_type
    )


class Config(BaseModel):
    """
    All config classes will inherit from this.
    Allows to homogenize model_config settings across all classes,
    and clarifies a bit naming standards (pydantic model_config vs our own).
    """

    model_config = get_config_dict()

    def update(self, **kwargs):
        self.__class__.validate(self.__dict__ | kwargs)
        self.__dict__.update(kwargs)
