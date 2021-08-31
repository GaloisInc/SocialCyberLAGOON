"""File which specifies plugin API.
"""

class Plugin:
    """Base plugin class. Provides a large number of hooks, all prefixed with
    `plugin_`. Those which are specified are considered supported capabilities.
    """

    def __init__(self, argument):
        pass


    @classmethod
    def plugin_caps(cls):
        """Returns a list of all capabilities (`plugin_` methods) supported by
        this plugin.
        """
        return [n for n in cls.__dict__ if n.startswith('plugin_')]


    def plugin_details_entity(self, entity):
        """Given some entity (with attributes loaded), return a raw HTML string
        to be rendered which adds some details to the UI.
        """
        raise NotImplementedError


    def plugin_details_observation(self, obs):
        """Given some observation (with entities + attributes loaded), return a
        raw HTML string to be rendered which adds some details to the UI.
        """
        raise NotImplementedError


    @classmethod
    def plugin_name(cls):
        return cls.__name__


    @classmethod
    def plugin_has(cls, name):
        # See if plugin has this specific method implemented
        if name in cls.__dict__:
            return True
        return False

