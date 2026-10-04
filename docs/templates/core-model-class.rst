{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {#- A model's parameters are listed in the "Parameters" section generated from its
       fields. They get no pages of their own: names that differ only in case (A_200,
       a_200) would clash. Only the model's metadata is listed here. #}
   {% set metadata = ["requires", "valid_domain", "calibration_domain",
                      "measured_mass_definition", "references", "parameter_source",
                      "normalized", "modifies_dndm", "alias"] %}
   {% block attributes %}
   {% if attributes %}
   .. rubric:: Model metadata
   {% for item in attributes if item in metadata %}

   .. autoattribute:: {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods
   {% for item in methods if item != "__init__" %}

   .. automethod:: {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
