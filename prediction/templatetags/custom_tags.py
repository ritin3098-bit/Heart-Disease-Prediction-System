from django import template
import sys

register = template.Library()

# Debug information
print("\n=== Loading custom_tags.py ===")
print(f"Python path: {sys.path}")
print(f"Module: {__name__}")

@register.filter(name='mul', is_safe=True)
def multiply(value, arg):
    """Multiply the value by the argument."""
    try:
        result = float(value) * float(arg)
        print(f"Multiplying {value} * {arg} = {result}")
        return result
    except (ValueError, TypeError) as e:
        print(f"Error in mul filter: {e}")
        try:
            result = value * arg
            print(f"Direct multiplication: {value} * {arg} = {result}")
            return result
        except Exception as e:
            print(f"Error in direct multiplication: {e}")
            return ''

# Verify the filter is registered
print(f"Registered filters: {register.filters.keys()}")
