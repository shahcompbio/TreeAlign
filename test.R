expr_simulated <- read.csv("data/expr_simulated.csv", stringsAsFactors = FALSE, row.names = 1)

expr <- read.csv("data/SPECTRUM-OV-022_expr_example.csv", stringsAsFactors = FALSE, row.names = 1)

expr_simulated <- t(expr_simulated)

expr_simulated <- expr_simulated/colSums(expr_simulated) * colSums(expr)


expr_mean <- rowSums(expr)


expected_expr <- read.csv("data/expected_expr_simulated.csv", stringsAsFactors = FALSE, row.names = 1)
expected_expr <- t(expected_expr)


expected_expr <- expected_expr / colSums(expected_expr)
