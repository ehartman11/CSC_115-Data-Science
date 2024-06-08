import nbformat as nbf


def py_to_ipynb(py_filename, ipynb_filename):
    with open(py_filename) as f:
        code = f.read()

    nb = nbf.v4.new_notebook()
    code_cells = [nbf.v4.new_code_cell(cell) for cell in code.split('\n\n')]
    nb['cells'] = code_cells

    with open(ipynb_filename, 'w') as f:
        nbf.write(nb, f)


py_to_ipynb('Assignment_1.1.py', 'Assignment_1.1.ipynb')
