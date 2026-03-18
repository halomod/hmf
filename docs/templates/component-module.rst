hmf.{{ objname }}
{{ underline }}

.. automodule:: {{ fullname }}

   {% block classes %}
   .. rubric:: Base Component(s)

   .. autosummary::
      :toctree: {{ objname }}
      :template: class.rst
   {% for item in classes %}
      {% if item.startswith("Base") or item.endswith("Component") %}
      {{ item }}
      {%- endif %}
   {%- endfor %}

   .. rubric:: Models

   .. autosummary::
      :toctree: {{ objname }}
      :template: class.rst
   {% for item in classes %}
      {% if not item.startswith("Base") and not item.endswith("Component") %}
      {{ item }}
      {%- endif %}
   {%- endfor %}

   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree: {{ objname }}
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
