cloneassign_prob <- read.csv("data/SPECTRUM-OV-022_clone_assign_prob_0.csv", stringsAsFactors = FALSE, row.names = 1)


cloneassign <- apply(cloneassign_prob, 1, which.max)

cloneassign_simulated <- read.csv("data/clone_assign_simulated_0.csv", stringsAsFactors = FALSE, row.names = 1)

cloneassign_simulated <- cloneassign_simulated$X0 + 1


gene_simulated <- read.csv("data/gene_type_score_simulated_0.csv", stringsAsFactors = FALSE, row.names = 1)
gene <- read.csv("data/SPECTRUM-OV-022_gene_type_score_0.csv", stringsAsFactors = FALSE, row.names = 1)


gene_simulated_assignment <- apply(gene_simulated, 1, which.max)
gene_assignment <- apply(gene, 1, which.max)

overlapping_genes <- intersect(names(gene_simulated_assignment), names(gene_assignment))

gene_simulated_assignment <- gene_simulated_assignment[overlapping_genes]
gene_assignment <- gene_assignment[overlapping_genes]
