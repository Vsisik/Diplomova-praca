import subprocess

# Generate documentation for your module or package using pdoc
# Replace 'your_module_or_package_name' with the actual name of your module or package
script = "dicomProcessing_helpingFunctions.py"
output_dir = "README"
output = "README_functions.html"

# Redirect the output of pdoc to a README file
subprocess.run(['pdoc', '-o ' + output_dir, script])

# subprocess.run(['pandoc', output_dir + '/' + script.replace(".py", ".html"), '-o', output.replace(".html", ".md")])
