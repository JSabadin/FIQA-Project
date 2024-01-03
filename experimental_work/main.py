from utils import *
import matplotlib.cm as cm
import csv
from pathlib import Path
import matplotlib.image as mpimg

def parse_dataset(dataset_name, pairs_file_path):
    # Parse the dataset based on the name
    if dataset_name == "xqlfw":
        return parse_verification_protocol_xqlfw(pairs_file_path)
    elif dataset_name == "cplfw":
        return parse_verification_protocol_cplfw(pairs_file_path)
    elif dataset_name == "adience":
        return parse_verification_protocol_adience(pairs_file_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
def calculate_quality_scores():
    models = [ "adaface", "arcface_o", "transface"]
    datasets = ["xqlfw", "cplfw", "adience"]

    perturbations = {
        "salt_and_pepper": {"s_vs_p": 0.5, "amount": 0.0002},
        "affine": {"rotation_angle": 0.5, "scale": 1.01, "flip_horizontal": True, "flip_vertical": False, "shear_x": 0, "shear_y": 0},
        "elastic": {"sigma": 1.0, "alpha": 3.0},
        "structured_occlusion": {"border_thickness": 1},
        "mixup": {"grid_size": (4, 4)}
        # Add more perturbations and their parameters as needed
    }

    for model_name in models:
        model = FaceEmbeddingModel(model_name)
        for dataset_name in datasets:
            dataset_path = os.path.join(project_root, 'datasets', dataset_name, 'images')
            pairs_file_path = os.path.join(project_root, 'datasets', dataset_name, f'{dataset_name}_pairs.txt')
            if dataset_name == "adience":
                verification_pairs = parse_dataset(dataset_name, dataset_path)
            else:
                verification_pairs = parse_dataset(dataset_name, pairs_file_path)
       
 
            # Extract unique filenames from verification pairs
            unique_filenames = {filename for pair in verification_pairs for filename in pair[:2]}

            for perturbation_name, params in perturbations.items():
                
                embeddings, quality_scores = process_dataset(dataset_path, model, perturbation_name, params, unique_filenames)

                # Save embeddings and quality scores
                embeddings_file = os.path.join(project_root, 'results', 'embeddings', f'{dataset_name}_{model_name}_embeddings.pkl')
                quality_scores_file = os.path.join(project_root, 'results', model_name, dataset_name, f'{dataset_name}_{perturbation_name}_quality_scores.pkl')
                
                save_results(embeddings_file, embeddings)
                save_results(quality_scores_file, quality_scores)

def calculate_pAUCs(discard_rates_fnmrs):
    pAUCs = {}
    discard_threshold = 0.3
    starting_error = next(iter(discard_rates_fnmrs.values()))[1][0]
    for label, (discard_rates, fnmrs) in discard_rates_fnmrs.items():
        discard_rates = np.array(discard_rates)
        fnmrs = np.array(fnmrs)
        fnmr_function = interp1d(discard_rates, fnmrs, kind='cubic', fill_value="extrapolate")
        valid_rates = discard_rates[discard_rates <= discard_threshold]
        partial_area_under_curve = np.trapz(fnmr_function(valid_rates) / starting_error, valid_rates)
        pAUCs[label] = partial_area_under_curve
    return pAUCs

def plot_all_EDC():
    models_raw = ["adaface", "arcface_o", "transface"]
    datasets_raw = ["xqlfw", "cplfw", "adience"]

    formatted_models = {"adaface": "AdaFace", "arcface_o": "ArcFace", "transface": "TransFace"}
    formatted_datasets = {"xqlfw": "XQLFW", "cplfw": "CPLFW", "adience": "Adience"}

    line_styles = ['-']

    # Define the "happy" colors (including the modified colors with one pink replaced by yellow)
    happy_colors = [
        "#32CD32",  # Lime Green
        "#FFD54F",  # Distinct Yellow
        "#FFB6C1",  # Light Pink
        "#00FFFF",  # Aqua
        "#0000FF",  # Blue
        "#8B3E00",  # Very Dark Orange
        "#FFA500",  # Orange
        "#FF6347",  # Tomato
        "#A52A2A",  # Brown
        "#DA70D6",  # Orchid
    ]


    # Create a custom cycler using the line styles and colors
    custom_cycler = cycler(linestyle=line_styles * (len(happy_colors) // len(line_styles))) * cycler(color=happy_colors)

    # Set the custom cycler to the current plot
    plt.rc('axes', prop_cycle=custom_cycler)

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    # Adjust the space between the subplots
    fig.subplots_adjust(hspace=0.3)  # Adjust this value as needed


    pAUC_results = []

    for i, model_name in enumerate(models_raw):
        for j, dataset_name in enumerate(datasets_raw):
            ax = axs[j, i]

            baselines_path = os.path.join(project_root, 'baselines')
            # results_path = os.path.join(project_root, 'results', model_name, dataset_name)
            results_path = os.path.join(project_root,  'results', "arcface_o", dataset_name)  # We use same quality scores for all!
            pkl_files = find_pkl_files(results_path, 'quality_scores') + find_pkl_files(baselines_path, dataset_name)

            path_emb = os.path.join(project_root, 'results', 'embeddings', f'{dataset_name}_{model_name}_embeddings.pkl')
            embeddings = load_pickle(path_emb)

            pairs_file_path = os.path.join(project_root, 'datasets', dataset_name, f'{dataset_name}_pairs.txt')
            if dataset_name == "adience":
                dataset_path = os.path.join(project_root, 'datasets', dataset_name, 'images')
                verification_pairs = parse_dataset(dataset_name, dataset_path)
            else:
                verification_pairs = parse_dataset(dataset_name, pairs_file_path)

            discard_rates_fnmrs = {}

            for file_path in pkl_files:
                file_name = os.path.basename(file_path)
                if "-quality" in file_name:
                    parent_directory = os.path.basename(os.path.dirname(file_path))
                    label = parent_directory
                else:
                    label = file_name

                scores_dict = load_pickle(file_path)
                new_scores_dict = {os.path.basename(f): v for f, v in scores_dict.items()}
                fnmrs, discard_rates = get_fnmrs(verification_pairs, embeddings, new_scores_dict)
                discard_rates_fnmrs[label] = (discard_rates, fnmrs)


            # Determine the maximum FNMR to display
            initial_fnmrs = [fnmrs[0] for _, (_, fnmrs) in discard_rates_fnmrs.items()]
            max_fnmr_to_display = max(initial_fnmrs) * 1.05  # Adjust the multiplier as needed
            for label, (discard_rates, fnmrs) in discard_rates_fnmrs.items():
                discard_rates = np.array(discard_rates)
                fnmrs = np.array(fnmrs)
                cleaned_label = label.replace('_quality_scores', '').replace('.pkl', '').replace('_', ' ').replace(f'{dataset_name}', 'FIQA ')
                ax.step(discard_rates, fnmrs, label=cleaned_label, alpha=0.7, linewidth=2, where='post')

            ax.set_xlabel('Fraction of discarded comparisons')
            ax.set_ylabel('FNMR')
            ax.grid(True)

            # Set the y-axis limit
            ax.set_ylim([None, max_fnmr_to_display])

            if i == 0:
                ax.annotate(formatted_datasets[dataset_name], xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center', fontweight='bold', rotation=90)
            if j == 0:
                axs[0, i].annotate(formatted_models[model_name], xy=(0.5, 1), xytext=(0, 5), 
                            xycoords='axes fraction', textcoords='offset points',
                            size='large', ha='center', va='baseline', fontweight='bold')
                
            # Calculate pAUCs for this model and dataset
            pAUCs = calculate_pAUCs(discard_rates_fnmrs)
            for label, pAUC in pAUCs.items():
                pAUC_results.append({
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Label": label,
                    "pAUC": pAUC
                })

    # Write pAUC results to a CSV file
    csv_filename = 'pAUC_results.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Model", "Dataset", "Label", "pAUC"])
        writer.writeheader()
        for result in pAUC_results:
            writer.writerow(result)
    print(f"pAUCs have been saved to {csv_filename}")

    handles, labels = axs[-1, -1].get_legend_handles_labels()
    def sort_labels(label):
        # We check if the label starts with 'FIQA'
        if label.startswith("FIQA"):
            # If it does, we return 0 which gives it highest priority
            return (0, label)
        else:
            # Otherwise, we return 1 which gives it lower priority
            return (1, label)

    sorted_handles_labels = sorted(zip(handles, labels), key=lambda t: sort_labels(t[1]))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)

    fig_legend = plt.figure(figsize=(12, 3))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(sorted_handles, sorted_labels, loc='center', ncol=5, mode='expand')
    ax_legend.axis('off')


    # Save the main figure
    filename_main = "EDC_all.png"  # Replace with your desired file name
    save_path_main = os.path.join('C:\\Users\\jerne\\Desktop\\FAKS\\MAG2\\Biometricni_sistemi\\BS_projekt\\slike', filename_main)
    fig.subplots_adjust(hspace=0.3, wspace=0.4)  # Adjust the wspace value as needed
    fig.savefig(save_path_main, format='png', dpi=600)

    # Save the legend figure
    filename_legend = "EDC_all_legend.png"  # Replace with your desired file name
    save_path_legend = os.path.join('C:\\Users\\jerne\\Desktop\\FAKS\\MAG2\\Biometricni_sistemi\\BS_projekt\\slike', filename_legend)
    fig_legend.savefig(save_path_legend, format='png', bbox_inches='tight', dpi=600)

    plt.close(fig)  # Close the main figure to free memory
    plt.close(fig_legend)  # Close the legend figure to free memory

def findQSimages():
    models = ['adaface', 'arcface_o', 'transface']
    formatted_models = {"adaface": "AdaFace", "arcface_o": "ArcFace", "transface": "TransFace"}
    num_images_per_model = 4  # Number of images per model to display

    base_results_path = Path(r"C:\Users\jerne\Desktop\FAKS\MAG2\Biometricni_sistemi\BS_projekt\experimental_tools\results")
    images_directory = Path(r"C:\Users\jerne\Desktop\FAKS\MAG2\Biometricni_sistemi\BS_projekt\experimental_tools\datasets\xqlfw\images")

    # Set up the figure and subplots
    fig, axs = plt.subplots(len(models), num_images_per_model, figsize=(6, 6))

    for model_idx, model in enumerate(models):
        pickle_file_path = base_results_path / model / 'xqlfw' / f'xqlfw_affine_quality_scores.pkl'

        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)

        image_names = list(data.keys())
        quality_scores = np.array(list(data.values()))

        sorted_indices = np.argsort(quality_scores)
        selected_indices = [sorted_indices[0], sorted_indices[len(sorted_indices) // 3], 
                            sorted_indices[2 * len(sorted_indices) // 3], sorted_indices[-1]]

        selected_images = [image_names[i] for i in selected_indices]
        selected_qs = [quality_scores[i] for i in selected_indices]

        def build_image_path(image_name):
            subject_name = "_".join(image_name.split('_')[:-1])
            return images_directory / subject_name / image_name

        for img_idx, (img_file, qs) in enumerate(zip(selected_images, selected_qs)):
            ax = axs[model_idx, img_idx]
            img_path = build_image_path(img_file)
            try:
                img = mpimg.imread(img_path)
                ax.imshow(img)
            except FileNotFoundError as e:
                print(e)
                ax.imshow(np.zeros((100, 100, 3)))
            ax.text(0.5, -0.15, f"QS: {qs:.2f}", transform=ax.transAxes, ha='center', fontsize=15, fontweight='bold')
            ax.axis('off')

        # Add model name on the left side of the row
        axs[model_idx, 0].text(-0.05, 0.5, formatted_models[model], transform=axs[model_idx, 0].transAxes, 
                               ha='right', va='center', fontsize=20, fontweight='bold', rotation=90)

    # Adjust subplot parameters to remove horizontal space between images
    plt.subplots_adjust(wspace=0, hspace=0)

    # Use tight_layout with minimal padding
    plt.tight_layout(pad=0)

    # Save the figure
    save_path = Path(r'C:\Users\jerne\Desktop\FAKS\MAG2\Biometricni_sistemi\BS_projekt\slike\QS_images.png')
    fig.savefig(save_path, format='png', dpi=600)

def plot_multiple_quality_scores_distribution(file_paths):
    """
    Reads multiple .pkl files containing dictionaries of image quality scores and plots their distributions in subplots,
    with the x-axis representing the quality score and the y-axis showing the relative frequency.

    Parameters:
    file_paths (list of str): A list of paths to the .pkl files.
    """
    # Number of files
    num_files = len(file_paths)

    # Create a figure with subplots
    fig, axs = plt.subplots(1, num_files, figsize=(15, 5), squeeze=False)
    axs = axs.flatten()  # Flatten the array to handle single subplot case
    titles = ["XQLFW", "CPLFW", "Adience"]

    for i, file_path in enumerate(file_paths):
        # Load the quality scores from the .pkl file
        with open(file_path, 'rb') as file:
            quality_scores = pickle.load(file)

        # Extract the scores from the dictionary
        scores = list(quality_scores.values())

        # Plot the distribution of quality scores in a subplot
        axs[i].hist(scores, bins=30, color='skyblue', edgecolor='black', density=True)
        axs[i].set_title(f'{titles[i]}', fontsize=14, fontweight='bold')
        axs[i].set_xlabel('Quality Score', fontsize=12, fontweight='bold')
        axs[i].set_ylabel('Relative Frequency [%]', fontsize=12, fontweight='bold')
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    calculate_quality_scores()
    plot_all_EDC()
    # plot_multiple_quality_scores_distribution(["C:\\Users\\jerne\\Desktop\\FAKS\\MAG2\\Biometricni_sistemi\\BS_projekt\\experimental_tools\\results\\arcface_o\\xqlfw\\xqlfw_affine_quality_scores.pkl", "C:\\Users\\jerne\\Desktop\\FAKS\\MAG2\\Biometricni_sistemi\\BS_projekt\\experimental_tools\\results\\arcface_o\\cplfw\\cplfw_affine_quality_scores.pkl", "C:\\Users\\jerne\\Desktop\\FAKS\\MAG2\\Biometricni_sistemi\\BS_projekt\\experimental_tools\\results\\arcface_o\\adience\\adience_affine_quality_scores.pkl"])