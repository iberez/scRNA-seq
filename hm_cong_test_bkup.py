
input_folder = '/bigdata/isaac/sd_amy_tsne_harmony_comp/iso/GABAint_sID_dataset/'
out_folder = '/bigdata/isaac/sd_amy_tsne_harmony_comp/iso/GABAint_sID_dataset/gaba_convergence_test/'
os.makedirs(out_folder, exist_ok=True)
os.makedirs(out_folder + 'data/', exist_ok=True)
iso_class_amy = 'GABA'

arr_pca  = np.load(input_folder + 'arr_pca_union_'+iso_class_amy+'.npy')
meta_data_df_pca = pd.read_json(input_folder + 'meta_data_df_pca_union_'+iso_class_amy+'.json')
amy_sd_hm_arr = np.load(input_folder + 'amy_sd_hm_arr_union_'+iso_class_amy+'.npy')
amy_sd_arr_tsne = np.load(input_folder + 'amy_sd_arr_tsne_union_'+iso_class_amy+'.npy')
vars_use = ['SampleID','dataset']

def run_harmony_convergence_test(input_folder, out_folder, iso_class_amy, arr_pca, meta_data_df_pca, amy_sd_hm_arr, amy_sd_arr_tsne, vars_use):
    x1 = hm.run_harmony(arr_pca,meta_data_df_pca,vars_use,max_iter_harmony=2)
    x2 = hm.run_harmony(x1.Z_corr.T,meta_data_df_pca,vars_use,max_iter_harmony=2)
    x3 = hm.run_harmony(x2.Z_corr.T,meta_data_df_pca,vars_use,max_iter_harmony=2)
    x4 = hm.run_harmony(x3.Z_corr.T,meta_data_df_pca,vars_use,max_iter_harmony=2)
    x5 = hm.run_harmony(x4.Z_corr.T,meta_data_df_pca,vars_use,max_iter_harmony=2)
    x6 = hm.run_harmony(x5.Z_corr.T,meta_data_df_pca,vars_use,max_iter_harmony=2)
    x7 = hm.run_harmony(x6.Z_corr.T,meta_data_df_pca,vars_use,max_iter_harmony=2)
    x8 = hm.run_harmony(x7.Z_corr.T,meta_data_df_pca,vars_use,max_iter_harmony=2)

    x0_pca = arr_pca
    x1_pca = x1.Z_corr.T
    x2_pca = x2.Z_corr.T
    x3_pca = x3.Z_corr.T
    x4_pca = x4.Z_corr.T
    x5_pca = x5.Z_corr.T
    x6_pca = x6.Z_corr.T
    x7_pca = x7.Z_corr.T
    x8_pca = x8.Z_corr.T

    pca_matrices = [x0_pca,x1_pca,x2_pca,x3_pca,x4_pca,x5_pca,x6_pca,x7_pca,x8_pca]
    #compute distances between consecutive pca matrices
    avg_distances = []
    for k in range(len(pca_matrices)-1):
        #difference between consecutive pca matrices
        diff = pca_matrices[k+1] - pca_matrices[k]
        #euclidean distance for each cell
        #print (diff.shape)
        distances = np.linalg.norm(diff, axis=1)
        #print (distances.shape)
        avg_distance = np.mean(distances)
        avg_distances.append(avg_distance)

    fig,ax = plt.subplots()
    ax.plot(range(1, len(pca_matrices)), avg_distances, marker='o')
    ax.set_xticks(range(1, len(pca_matrices)),labels = [f'{2*i}' for i in range(1,len(pca_matrices))])
    #ax.set_xticklabels(range(1, len(pca_matrices)), labels = [f'{2*i}-{2*i+1}' for i in range(len(pca_matrices)-1)])
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Distance')
    ax.set_title('Average Distance Between Consecutive PCA Matrices')
    plt.savefig(out_folder + 'avg_distances_harmony_convergence.pdf')
    plt.show()

    for i,m in enumerate(pca_matrices):
        tsne_plot(m,i,meta_data_df_pca,out_folder,savefig=True, write_to_file=True)

    files = [x for x in os.listdir(out_folder) if 'iter' in x]
    # Create a PdfFileWriter object to write the final PDF
    output_pdf = PdfWriter()

    # Create a 2x2 grid of the plots
    fig, axs = plt.subplots(2, 4, figsize=(20, 20))

    # Read and add each PDF file to the grid
    for ax, pdf_file in zip(axs.flat, files):
        pdf_path = os.path.join(out_folder, pdf_file)

        # Convert the first page of the PDF to an image
        images = convert_from_path(pdf_path, first_page=0, last_page=1)
        image = images[0]

        # Display the image in the grid
        #fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image, aspect = 'equal')
        ax.axis('off')
        ax.set_title(pdf_file)


    # Save the final PDF
    output_pdf_path = os.path.join(out_folder, 'hm_stiched.pdf')
    with PdfPages(output_pdf_path) as pdf:
        pdf.savefig(fig)

    print(f"Stitched PDF saved as {output_pdf_path}")
    # Create a PdfFileWriter object to write the final PDF
    output_pdf = PdfWriter()

    # Create a 2x2 grid of the plots
    fig, axs = plt.subplots(2, 4, figsize=(20, 20))

    # Read and add each PDF file to the grid
    for ax, pdf_file in zip(axs.flat, files):
        pdf_path = os.path.join(out_folder, pdf_file)

        # Convert the first page of the PDF to an image
        images = convert_from_path(pdf_path, first_page=0, last_page=1)
        image = images[0]

        # Display the image in the grid
        #fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image, aspect = 'equal')
        ax.axis('off')
        ax.set_title(pdf_file)


    # Save the final PDF
    output_pdf_path = os.path.join(out_folder, 'hm_stiched_iter.pdf')
    with PdfPages(output_pdf_path) as pdf:
        pdf.savefig(fig)

    print(f"Stitched PDF saved as {output_pdf_path}")
    
    return None