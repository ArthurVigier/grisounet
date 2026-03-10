def linreg(X,y):
    model = LinearRegression(tol=1)
    model = model.fit(X, y)
    print('R2: ', r2_score(y, model.predict(X)))
    return model


