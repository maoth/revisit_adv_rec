    def validate(self, train_data, test_data, target_items):
        """Evaluate attack performance on target items."""
        t1 = time.time()

        n_rows = train_data.shape[0]
        n_evaluate_users = test_data.shape[0]

        # Init evaluation results.
        target_items_position = np.zeros([n_rows, len(target_items)], dtype=np.int64)

        recommendations = self.recommend(train_data, top_k=100)

        top_1 = self.recommend(train_data, top_k=1)
        top_1 = [a[0] for a in top_1]

        rows = np.array(range(len(top_1)))
        cols = np.array(top_1)
        data = np.ones_like(rows)

        B = coo_matrix((data, (rows, cols)), shape=train_data.shape)
        new_train_data = train_data + B.tocsr()

        # print(train_data.nnz)
        # print(new_train_data.nnz)

        new_recommendations = self.recommend(new_train_data, top_k=100)
        
        valid_rows = list()
        for i in range(train_data.shape[0]):
            # Ignore augmented users, evaluate only on real users.
            if i >= n_evaluate_users:
                continue
            targets = test_data[i].indices
            
            if targets.size <= 0:
                continue

            recs = recommendations[i].tolist()

            for j, item in enumerate(target_items):
                if item in recs:
                    target_items_position[i, j] = recs.index(item)
                else:
                    target_items_position[i, j] = train_data.shape[1]

            valid_rows.append(i)
        target_items_position_1 = target_items_position[valid_rows]

        valid_rows = list()
        for i in range(train_data.shape[0]):
            # Ignore augmented users, evaluate only on real users.
            if i >= n_evaluate_users:
                continue
            targets = test_data[i].indices
            
            if targets.size <= 0:
                continue

            recs = new_recommendations[i].tolist()

            for j, item in enumerate(target_items):
                if item in recs:
                    target_items_position[i, j] = recs.index(item)
                else:
                    target_items_position[i, j] = train_data.shape[1]

            valid_rows.append(i)
        target_items_position_2 = target_items_position[valid_rows]


        # Summary evaluation results into a dict.
        result = OrderedDict()
        result["TargetAvgRank"] = (target_items_position_1.mean() + target_items_position_2.mean()) / 2
        # Note that here target_pos starts from 0.
        cutoff = 20
        result["TargetHR@%d" % cutoff] = (np.logical_or(target_items_position_1 < cutoff, target_items_position_2 < cutoff).sum(1) >= 1).mean()

        # Log results.
        print("Attack Evaluation [{:.1f} s], {} ".format(
            time.time() - t1, str(result)))
        return result
