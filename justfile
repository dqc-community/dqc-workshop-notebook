# Usage:
#   just mo tutorial.py
#   just mo
mo notebook="tutorial.py":
  uv run marimo edit {{notebook}} --sandbox

# Usage:
#   just mcp notebook.py
#   just mcp notebook.py 2718
#   just mcp
mcp notebook="tutorial.py" port="2718":
  uv run marimo edit {{notebook}} --mcp --no-token --no-sandbox --port {{port}}
