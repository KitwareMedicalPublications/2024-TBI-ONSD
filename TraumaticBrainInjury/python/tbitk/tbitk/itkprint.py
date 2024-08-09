import itk

# TODO: maybe make this interoperable with python print()
def vnl_matrix_to_nparray(x):
    ans = []
    nr = y.rows()
    nc = y.cols()
    for i in range(0, nr):
        row = []
        for j in range(0, nc):
            row.append(x.get(i,j))
        ans.append(row)
    return np.array(ans)

            

def itkprint(x):
    '''
    Print or display an ITK object appropriately for a Jupyter Lab notebook
    '''
    y = str(type(x))
    if y.startswith("<class 'itkMatrixPython.itkMatrix"):
        print(vnl_matrix_to_nparray(x.GetVnlMatrix()))
    elif y.startswith("<class 'itkOptimizerParametersPython.itkOptimizerParameters"):
        print([x.GetElement(i) for i in range(0, x.GetSize())])
    else:
        print(x)