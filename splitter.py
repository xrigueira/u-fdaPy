import pandas as pd

"""This function splits the data processed into
different files, one for each variable."""

#NEEDSWORK: WET code.

def splitter(File):

    # Read the file and store it as a pandas database
    df = pd.read_csv(f'Database/{File}', delimiter=';', parse_dates=['date'], index_col=['date'])

    dfAmonio = df[['ammonium', 'year', 'month', 'day', 'hour', 'minute', 'second', 'week', 'startDate', 'endDate', 'weekOrder']]
    dfConductividad = df[['conductivity', 'year', 'month', 'day', 'hour', 'minute', 'second', 'week', 'startDate', 'endDate', 'weekOrder']]
    dfNitratos = df[['nitrates', 'year', 'month', 'day', 'hour', 'minute', 'second', 'week', 'startDate', 'endDate', 'weekOrder']]
    dfOxigeno = df[['oxygen', 'year', 'month', 'day', 'hour', 'minute', 'second', 'week', 'startDate', 'endDate', 'weekOrder']]
    dfph = df[['pH', 'year', 'month', 'day', 'hour', 'minute', 'second', 'week', 'startDate', 'endDate', 'weekOrder']]
    dfTemperatura = df[['temperature', 'year', 'month', 'day', 'hour', 'minute', 'second', 'week', 'startDate', 'endDate', 'weekOrder']]
    dfTurbidez = df[['turbidity', 'year', 'month', 'day', 'hour', 'minute', 'second', 'week', 'startDate', 'endDate', 'weekOrder']]
    dfCaudal = df[['flow', 'year', 'month', 'day', 'hour', 'minute', 'second', 'week', 'startDate', 'endDate', 'weekOrder']]
    dfPluviometria = df[['pluviometry', 'year', 'month', 'day', 'hour', 'minute', 'second', 'week', 'startDate', 'endDate', 'weekOrder']]

    colsAmonio = list(dfAmonio.columns.values.tolist())
    colsAmonio[0] = 'value'
    dfAmonio.to_csv(f'Database/Amonio_pro.csv', sep=';', encoding='utf-8', index=True, header=colsAmonio)

    colsConductividad = list(dfConductividad.columns.values.tolist())
    colsConductividad[0] = 'value'
    dfConductividad.to_csv(f'Database/Conductividad_pro.csv', sep=';', encoding='utf-8', index=True, header=colsConductividad)

    colsNitratos = list(dfNitratos.columns.values.tolist())
    colsNitratos[0] = 'value'
    dfNitratos.to_csv(f'Database/Nitratos_pro.csv', sep=';', encoding='utf-8', index=True, header=colsNitratos)

    colsOxigeno = list(dfOxigeno.columns.values.tolist())
    colsOxigeno[0] = 'value'
    dfNitratos.to_csv(f'Database/Oxigeno disuelto_pro.csv', sep=';', encoding='utf-8', index=True, header=colsOxigeno)

    colsph = list(dfph.columns.values.tolist())
    colsph[0] = 'value'
    dfph.to_csv(f'Database/pH_pro.csv', sep=';', encoding='utf-8', index=True, header=colsph)

    colsTemperatura = list(dfTemperatura.columns.values.tolist())
    colsTemperatura[0] = 'value'
    dfTemperatura.to_csv(f'Database/Temperatura_pro.csv', sep=';', encoding='utf-8', index=True, header=colsTemperatura)

    colsTurbidez = list(dfTurbidez.columns.values.tolist())
    colsTurbidez[0] = 'value'
    dfTurbidez.to_csv(f'Database/Turbidez_pro.csv', sep=';', encoding='utf-8', index=True, header=colsTurbidez)

    colsCaudal = list(dfCaudal.columns.values.tolist())
    colsCaudal[0] = 'value'
    dfCaudal.to_csv(f'Database/Caudal_pro.csv', sep=';', encoding='utf-8', index=True, header=colsCaudal)

    colsPluviometria = list(dfPluviometria.columns.values.tolist())
    colsPluviometria[0] = 'value'
    dfPluviometria.to_csv(f'Database/Pluviometria_pro.csv', sep=';', encoding='utf-8', index=True, header=colsPluviometria)
