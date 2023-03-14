import numpy as np
import pandas as pd
import pysr
from sklearn.model_selection import train_test_split
from pysr import PySRRegressor


def giveData(Re, phi, spherical, n_neighbors, node_update):
    filename = f'Re{Re.replace(".","")}_phi{phi.replace(".","")}'
    if spherical == True:
        filename += '_spherical'
        columns = ['r', 'theta', 'phi', 'u_x', 'u_y', 'u_z']
    else:
        filename += '_cartesian'
        columns = ['x', 'y', 'z' , 'u_x', 'u_y', 'u_z']
    if node_update == True:
        filename += '_nodeupdate'
   
    dataset = pd.read_csv('./GP_data/'+filename+'.csv')

    X = dataset[columns]
    y = dataset[['F_i', 'F_true']]

    # Half of the data samples for training, other half for testing
    n_arrangements = y.shape[0]//n_neighbors
    n_train = n_arrangements//2*n_neighbors
    n_test = y.shape[0]-n_train
    X_train, X_test, _y_train, _y_test = train_test_split(X, y, shuffle=False, train_size=n_train)
    y_train = _y_train['F_i']
    y_test = _y_test['F_i']
    F_test = _y_test['F_true'].to_numpy()
    F_test = np.array([F_test[i*n_neighbors] for i in range(n_test//n_neighbors)])
    
    #TODO: Only use training data during fitting. Test data shall generate an R^2 plot. 
    F_nodeupdate_train = _y_train['F_true'].to_numpy()
    F_nodeupdate_train = np.array([F_nodeupdate_train[i*n_neighbors] for i in range(F_nodeupdate_train.shape[0]//n_neighbors)])
    X_nodeupdate_train = X_train[['u_x', 'u_y', 'u_z']].to_numpy()
    X_nodeupdate_train = np.array([X_nodeupdate_train[i*n_neighbors] for i in range(X_nodeupdate_train.shape[0]//n_neighbors)])
    X_nodeupdate_test = X_test[['u_x', 'u_y', 'u_z']].to_numpy()
    X_nodeupdate_test = np.array([X_nodeupdate_test[i*n_neighbors] for i in range(X_nodeupdate_test.shape[0]//n_neighbors)])

    return X_train, y_train, F_nodeupdate_train, X_nodeupdate_train, X_test, y_test, F_test, X_nodeupdate_test, filename

def runGP(X_train, y_train, X_nodeupdate_train, F_nodeupdate_train, n_iterations, run, const_complexity, model_selection, filename, node_update, n_neighbors):

    #Define algorithm settings
    bin_operators       = ["+", "*", "/"]
    un_operators        = ["cos", "sin", "inv(x) = 1/x", "tan", "exp", "log"]
    comp_of_operators   = {"+": 1, "*": 1, "/": 1, "inv":1, "sin": 2, "cos": 2, "tan": 2, "log": 2, "exp": 2}
    comp_of_var         = 1
    constr              = {'cos': 7, 'sin': 7, 'tan': 7, 'log': 7}
    nested_constr       = {"cos": {"cos": 0}, "sin": {"sin": 0}, "tan": {"tan": 0}, "sin": {"tan": 0}, "cos": {"tan": 0}, "tan": {"cos": 0}, "tan": {"sin": 0}, "cos": {"sin": 0}, "sin": {"cos": 0}}

    if node_update == False:
        if const_complexity == 1:
            filename += '_comp1'
        model = pysr.PySRRegressor(
            model_selection=model_selection, 
            population_size = 100,
            niterations=n_iterations,
            binary_operators=bin_operators,
            unary_operators=un_operators,
            extra_sympy_mappings={"inv": lambda x: 1/x},
            equation_file='./GP_results/'+filename+'_equations_'+str(run)+'.csv',
            constraints=constr,
            nested_constraints=nested_constr,
            verbosity=1,
            complexity_of_operators=comp_of_operators,
            complexity_of_constants=const_complexity,
            complexity_of_variables=comp_of_var,
            )
        model.fit(X_train, y_train)
    
    elif node_update == True:
        msg_model = pysr.PySRRegressor(
            model_selection=model_selection, 
            population_size = 100,
            niterations=n_iterations,
            binary_operators=bin_operators,
            unary_operators=un_operators,
            extra_sympy_mappings={"inv": lambda x: 1/x},
            equation_file='./GP_results/'+filename+'_msg_equations_'+model_selection+'_'+str(run)+'.csv',
            constraints=constr,
            nested_constraints=nested_constr,
            verbosity=1,
            complexity_of_operators=comp_of_operators,
            complexity_of_constants=const_complexity,
            complexity_of_variables=comp_of_var,
        )
        msg_model.fit(X_train, y_train)

        node_model = pysr.PySRRegressor(
            model_selection=model_selection,
            population_size = 100,
            niterations=n_iterations,
            binary_operators=bin_operators,
            unary_operators=un_operators,
            extra_sympy_mappings={"inv": lambda x: 1/x},
            equation_file='./GP_results/'+filename+'_node_equations_'+model_selection+'_'+str(run)+'.csv',
            constraints=constr,
            nested_constraints=nested_constr,
            verbosity=1,
            complexity_of_operators=comp_of_operators,
            complexity_of_constants=const_complexity,
            complexity_of_variables=comp_of_var,
        )

        msg_model = PySRRegressor.from_file('./GP_results/'+filename+'_msg_equations_'+model_selection+'_'+str(run)+'.csv', model_selection = model_selection)
        msg_model.set_params(extra_sympy_mappings={'inv': lambda x: 1/x})
        msg_model.refresh()

        y_msg_pred = msg_model.predict(X_train)
        F_msg_pred = np.array([np.sum(y_msg_pred[i*n_neighbors:(i+1)*n_neighbors]) for i in range(y_msg_pred.shape[0]//n_neighbors)])
        X_n = np.concatenate((F_msg_pred.reshape((F_msg_pred.shape[0],1)), X_nodeupdate_train), axis = 1)
        X_n = pd.DataFrame(X_n, columns = ['F_i', 'u_x', 'u_y', 'u_z'])
        
        node_model.fit(X_n, F_nodeupdate_train)


def R2(y_pred, y_true):
    err =  np.sum(np.square(y_true - y_pred)) 
    mean = np.sum(np.square(y_true - np.mean(y_true))) 
    return 1 - err / (mean + 1e-30) # add tiny value 1e-30 to avoid division by 0

def MSE(y_pred, y_true):
    return np.mean(np.square(y_true - y_pred))

if __name__ == '__main__':
    # Number of current run
    run = 0
    #So far, we provide datasets for phi in {0.064, 0.125, 0.216, 0.343}
    Re = '0'
    phi = '0.064'
    #Set to True if spherical coordinates shall be used (This is the standard setting of our experiments)
    spherical = True
    #The provided datasets include 30 particles, of which one is in located in the center and surrounded by 29 neighbors 
    n_particles = 30
    n_neighbors = n_particles - 1
    #If node_update == True: y = f(g(x)), if node_update == False: y = f(x)
    node_update = True
    #const_complexity is either 1 or 2, only relevant if node_update == False
    const_complexity = 1
    #For test, set to true and only 2 iterations will be completed
    test = False
    n_iterations = 100 if test == False else 2
    #Choose between 'accuracy' and 'best' (See PySR documentation for more information on model selection). We used 'best' in this paper. 
    model_selection = 'best'

    X_train, y_train, F_nodeupdate_train, X_nodeupdate_train, X_test, y_test, F_test, X_nodeupdate_test, filename = giveData(Re, phi, spherical, n_neighbors, node_update)
    #Comment the next line if only analysis of a previously trained model is required
    runGP(X_train, y_train, X_nodeupdate_train, F_nodeupdate_train, n_iterations, run, const_complexity, model_selection, filename, node_update, n_neighbors)    
    
    
    if (node_update == False) and (const_complexity == 1):
        filename += '_comp1'
    if node_update == False:
        model = PySRRegressor.from_file('./GP_results/'+filename+'_equations_'+str(run)+'.pkl', model_selection = 'best')
        model.set_params(extra_sympy_mappings={'inv': lambda x: 1/x})
        model.refresh()

        y_test_pred = model.predict(X_test)
        F_test_pred = np.array([np.sum(y_test_pred[i*n_neighbors:(i+1)*n_neighbors]) for i in range(y_test_pred.shape[0]//n_neighbors)])
        print('---- Test results for y = f(x) ----')
        print('Test R^2: ', R2(F_test_pred, F_test))
        print('Test MSE: ', MSE(F_test_pred, F_test))

    if node_update == True:
        msg_model = PySRRegressor.from_file('./GP_results/'+filename+'_msg_equations_'+model_selection+'_'+str(run)+'.pkl', model_selection = model_selection)
        node_model = PySRRegressor.from_file('./GP_results/'+filename+'_node_equations_'+model_selection+'_'+str(run)+'.pkl', model_selection = model_selection)
        msg_model.set_params(extra_sympy_mappings={'inv': lambda x: 1/x})
        msg_model.refresh()
        node_model.set_params(extra_sympy_mappings={'inv': lambda x: 1/x})
        node_model.refresh()

        y_test_msg_pred = msg_model.predict(X_test)
        F_test_msg_pred = np.array([np.sum(y_test_msg_pred[i*n_neighbors:(i+1)*n_neighbors]) for i in range(y_test_msg_pred.shape[0]//n_neighbors)])
        X_n = np.concatenate((F_test_msg_pred.reshape((F_test_msg_pred.shape[0],1)), X_nodeupdate_test), axis = 1)
        X_n = pd.DataFrame(X_n, columns = ['F_i', 'u_x', 'u_y', 'u_z'])
        F_test_node_pred = node_model.predict(X_n)
        print('---- Test results for y = f(g(x)) ----')
        print('Test R^2: ', R2(F_test_node_pred, F_test))
        print('Test MSE: ', MSE(F_test_node_pred, F_test))

        #Finally, the expert chooses one of the equations from the Pareto front in the equation file (or leave it to the algorithm by setting model_selection = 'best').
        #Don't forget to refit the constants to the original dataset, since we trained the symbolic model on the output of the graph network.