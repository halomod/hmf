{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes
   {#- Inline, not a page per attribute: ~1000 such pages made the build quadratic
       in the number of pages (each page renders the full navigation tree). #}
   {% for item in attributes %}

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
