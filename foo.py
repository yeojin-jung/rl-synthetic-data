
    def select_g_vendi(train_grads, k):
        """
        G-Vendi: Greedy Diversity Selection.
        Keeps original logic: Re-calculates full Vendi score for every candidate.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Keep master tensor on GPU
        train_grads = train_grads.to(device)
        
        n = train_grads.shape[0]
        selected_indices = []
        remaining_indices = list(range(n))

        for _ in tqdm(range(k), desc="G-Vendi Greedy"):
            best_vendi = -1
            best_idx = -1

            for idx in remaining_indices:
                current_set = selected_indices + [idx]
                # Slicing on GPU
                grads_subset = train_grads[current_set]
                
                gv = get_g_vendi(grads_subset, len(current_set))

                if gv > best_vendi:
                    best_vendi = gv
                    best_idx = idx
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return np.array(selected_indices)