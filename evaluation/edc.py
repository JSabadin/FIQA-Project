import numpy as np


def distance_(embeddings0, embeddings1):
    # Distance based on cosine similarity
    dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
    norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
    # shaving
    similarity = np.clip(dot / norm, -1., 1.)
    dist = np.arccos(similarity) / np.pi
    return dist


def calc_score(embeddings0, embeddings1, actual_issame, subtract_mean=False):
    assert(embeddings0.shape[0] == embeddings1.shape[0])
    assert(embeddings0.shape[1] == embeddings1.shape[1])

    if subtract_mean:
        mean = np.mean(np.concatenate([embeddings0, embeddings1]), axis=0)
    else:
        mean = 0.
    dist = distance_(embeddings0, embeddings1)
    # sort in a desending order
    pos_scores = np.sort(dist[actual_issame==1])
    neg_scores = np.sort(dist[actual_issame==0])
    return pos_scores, neg_scores


def get_fnmrs(dataset, embeddings, quality_scores, false_match_rate=1e-3):
    """Calculates the EDC curve values given a verification dataset protocol, dataset embeddings and quality scores.

    Args:
        dataset (list): Verification pairs in the form (image1, image2, is_genuine). is_genuine is 1 for positive pairs and 0 for negative pairs.
        embeddings (dict): Embeddings in the form {image1: embedding}
        quality_scores (dict): Quality scores in the form {image1: quality_score}
        false_match_rate (float, optional): False match rate for which to calculate the False Non Match Rates. Defaults to 1e-3.

    Returns:
        (list, list): Returns a list of FNMR values and the corresponding Discard rates.
    """

    embeddings1, embeddings2, is_genuine, quality = [], [], [], []

    for (name1, name2, label) in dataset:

        embeddings1.append(embeddings[name1])
        embeddings2.append(embeddings[name2])
        is_genuine.append(label)

        quality.append(min(quality_scores[name1], quality_scores[name2]))

    embeddings1 = np.vstack(embeddings1)
    embeddings2 = np.vstack(embeddings2)
    is_genuine  = np.vstack(is_genuine).reshape(-1,)
    quality     = np.array(quality)
    quality_id  = np.argsort(quality)

    num_pairs = len(is_genuine)
    unconsidered_rates = np.arange(0, 0.98, 0.05)
    fnmrs_list = []
    used_ur = []
    for u_r in unconsidered_rates:

        hq_pairs_idx = quality_id[int(u_r*num_pairs):]

        pos_dists, neg_dists = calc_score(embeddings1[hq_pairs_idx], embeddings2[hq_pairs_idx], is_genuine[hq_pairs_idx])

        fmr = 1
        idx = len(neg_dists) - 1
        num_query = len(pos_dists)
        while idx >= 0:
            thresh = neg_dists[idx]
            num_acc = sum(pos_dists < thresh)
            fnmr = 1.0 * float(num_query - num_acc) / float(num_query) if num_query != 0 else 0.

            if fmr == false_match_rate:
                sym = ' ' if thresh >= 0 else ''
                line = 'FNMR = {:.10f}  :  FMR = {:.10f}  :  THRESHOLD = {}{:.10f}  :   DROP.RATE = {:.5f}'.format(fnmr, fmr, sym, thresh, u_r)
                print(line)
                fnmrs_list.append(fnmr)
                used_ur.append(u_r)
                break

            if idx == 0:
                break
            idx /= 10
            idx = int(idx)
            fmr /= float(10) 

    return fnmrs_list, used_ur