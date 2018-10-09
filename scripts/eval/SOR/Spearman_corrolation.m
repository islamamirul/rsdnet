% compute Spearman’s Rank-Order Correlation between the two inputs

function RHO_Spearman = Spearman_corrolation(vector_1, vector_2)
vector_1 = tiedrank(vector_1');
vector_2 = tiedrank(vector_2');

RHO_Spearman = (corr(vector_1(:), vector_2(:), 'Type', 'Spearman'));



