import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go


# Plotly
def plotly_pie(df, values, names, title):
    fig = px.pie(df, values=values, names=names, title=title)
    fig.update_traces(textposition='inside', textinfo='label+percent')
    # fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    fig.show()


def plt_pie(data, labels, title):
    plt.figure(figsize=(20, 18))
    figure_object, axes_object = plt.subplots()
    plt.title(title)
    axes_object.pie(data,
                    labels=labels,
                    shadow=True,
                    autopct='%.1f%%',
                    # wedgeprops={'linewidth': 3, 'edgecolor': "orange"}
                    )
    axes_object.axis('equal')
    plt.show()


# création d'une fonction pour la visualisation de chaque variable selon son type
def plot_var(df, col_name, full_name, continuous):
    """
    visualisation d'une variable avec ou sans la fonction faceting en lien avec la target(v.a classification).
    - col_name est le nom de la variable dans le data frame
    - full_name est le nom attribué à la variable dans la figure
    - continuous si vrai (true), la variable est continue
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, figsize=(12, 10))

    # plot1: compte la distribution de la variable target sur les variables continues

    if continuous:
        sns.distplot(df.loc[df[col_name].notnull(), col_name], kde=False, ax=ax1)
    else:
        sns.countplot(df[col_name], order=sorted(df[col_name].unique()), color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(full_name)
    ax1.set_ylabel('Count')
    ax1.set_title(full_name)

    # plot2: bar plot of the variable grouped by classification
    if continuous:
        sns.boxplot(x=col_name, y='classification', data=df, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(full_name + ' Par Classification')
    else:
        classification_grouped = df.groupby(col_name)['classification'].value_counts(normalize=True)
        sns.barplot(x=classification_grouped.index, y=classification_grouped.values, color='#5975A4', saturation=1,
                    ax=ax2)
        ax2.set_ylabel('Fraction de la classification')
        ax2.set_title('Taux de classification par ' + full_name)
        ax2.set_xlabel(full_name)

    # plot2: kde plot des variables groupées par la variable target
    if continuous:
        facet = sns.FacetGrid(df, hue='classification', size=3, aspect=4)
        facet.map(sns.kdeplot, col_name, shade=True)
        # facet.set(xlim=(df[col_name].min(), df[col_name].max()))
        facet.add_legend()
    else:
        fig = plt.figure(figsize=(12, 3))
        sns.countplot(x=col_name, hue='classification', data=df, order=sorted(df[col_name].unique()))

    plt.tight_layout()
