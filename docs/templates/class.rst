{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree: {{ objname }}
      :noindex:

   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}


   {% block methods %}

   {% if methods %}
   .. rubric:: Methods

   {% for item in methods %}
   .. automethod:: {{ item }}
      :noindex:
   {% endfor %}

   {% endif %}
   {% endblock %}
