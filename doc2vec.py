# -*- coding: utf-8 -*-
# Doc2Vec Model
#---------------------------------------
#
# In this example, we will download and preprocess the movie
#  review data.
#
# From this data set we will compute/fit a Doc2Vec model to get
# Document vectors.  From these document vectors, we will split the
# documents into train/test and use these doc vectors to do sentiment
# analysis on the movie review dataset.

def doc2vecRun(window_size = 3, embedding_size = 64, dataName = 'slantour_data.txt'):
    import tensorflow as tf
    import numpy as np
    import random
    import os
    import pickle
    import text_helpers
    from tensorflow.python.framework import ops
    ops.reset_default_graph()

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Make a saving directory if it doesn't exist
    data_folder_name = 'data'
    if not os.path.exists(data_folder_name):
        os.makedirs(data_folder_name)

    # Start a graph session
    sess = tf.Session()

    # Declare model parameters
    batch_size = 32
    vocabulary_size = 7500
    generations = 500000
    model_learning_rate = 0.1

    #embedding_size = 64   # Word embedding size
    doc_embedding_size = embedding_size   # Document embedding size
    concatenated_size = embedding_size + doc_embedding_size

    num_sampled = int(batch_size/2)    # Number of negative examples to sample.
    #window_size = 3       # How many words to consider to the left.

    # Add checkpoints to training
    save_embeddings_every = 50000
    print_valid_every = 50000
    print_loss_every = 1000

    # Declare stop words
    #stops = stopwords.words('english')
    stops = ["a","aby","ahoj","aj","ale","anebo","ani","aniž","ano","asi","aspoåˆ","aspoň","atd","atp","az","aäkoli","ačkoli","až","bez","beze","blã­zko","blízko","bohuå¾el","bohužel","brzo","bude","budem","budeme","budes","budete","budeå¡","budeš","budou","budu","by","byl","byla","byli","bylo","byly","bys","byt","bä›hem","být","během","chce","chceme","chcete","chceå¡","chceš","chci","chtã­t","chtä›jã­","chtít","chtějí","chut'","chuti","ci","clanek","clanku","clanky","co","coz","což","cz","daleko","dalsi","další","den","deset","design","devatenáct","devatenã¡ct","devä›t","devět","dnes","do","dobrã½","dobrý","docela","dva","dvacet","dvanáct","dvanã¡ct","dvä›","dvě","dál","dále","dã¡l","dã¡le","dä›kovat","dä›kujeme","dä›kuji","děkovat","děkujeme","děkuji","email","ho","hodnä›","hodně","i","jak","jakmile","jako","jakož","jde","je","jeden","jedenáct","jedenã¡ct","jedna","jedno","jednou","jedou","jeho","jehož","jej","jeji","jejich","jejã­","její","jelikož","jemu","jen","jenom","jenž","jeste","jestli","jestliå¾e","jestliže","jeå¡tä›","ještě","jež","ji","jich","jimi","jinak","jine","jiné","jiz","již","jsem","jses","jseš","jsi","jsme","jsou","jste","já","jã¡","jã­","jã­m","jí","jím","jíž","jšte","k","kam","každý","kde","kdo","kdy","kdyz","kdyå¾","když","ke","kolik","kromä›","kromě","ktera","ktere","kteri","kterou","ktery","která","kterã¡","kterã©","kterã½","které","který","kteå™ã­","kteři","kteří","ku","kvå¯li","kvůli","ma","majã­","mají","mate","me","mezi","mi","mit","mne","mnou","mnä›","mně","moc","mohl","mohou","moje","moji","moå¾nã¡","možná","muj","musã­","musí","muze","my","má","málo","mám","máme","máte","máš","mã¡","mã¡lo","mã¡m","mã¡me","mã¡te","mã¡å¡","mã©","mã­","mã­t","mä›","må¯j","må¯å¾e","mé","mí","mít","mě","můj","může","na","nad","nade","nam","napiste","napište","naproti","nas","nasi","naå¡e","naå¡i","načež","naše","naši","ne","nebo","nebyl","nebyla","nebyli","nebyly","nechť","nedä›lajã­","nedä›lã¡","nedä›lã¡m","nedä›lã¡me","nedä›lã¡te","nedä›lã¡å¡","nedělají","nedělá","nedělám","neděláme","neděláte","neděláš","neg","nejsi","nejsou","nemajã­","nemají","nemáme","nemáte","nemã¡me","nemã¡te","nemä›l","neměl","neni","nenã­","není","nestaäã­","nestačí","nevadã­","nevadí","nez","neå¾","než","nic","nich","nimi","nove","novy","nové","nový","nula","ná","nám","námi","nás","náš","nã¡m","nã¡mi","nã¡s","nã¡å¡","nã­m","nä›","nä›co","nä›jak","nä›kde","nä›kdo","nä›mu","ní","ním","ně","něco","nějak","někde","někdo","němu","němuž","o","od","ode","on","ona","oni","ono","ony","osm","osmnáct","osmnã¡ct","pak","patnáct","patnã¡ct","po","pod","podle","pokud","potom","pouze","pozdä›","pozdě","poå™ã¡d","pořád","prave","pravé","pred","pres","pri","pro","proc","prostä›","prostě","prosã­m","prosím","proti","proto","protoze","protoå¾e","protože","proä","proč","prvni","první","práve","pta","pä›t","på™ed","på™es","på™ese","pět","před","přede","přes","přese","při","přičemž","re","rovnä›","rovně","s","se","sedm","sedmnáct","sedmnã¡ct","si","sice","skoro","smã­","smä›jã­","smí","smějí","snad","spolu","sta","sto","strana","stã©","sté","sve","svych","svym","svymi","své","svých","svým","svými","svůj","ta","tady","tak","take","takhle","taky","takze","také","takže","tam","tamhle","tamhleto","tamto","tato","te","tebe","tebou","ted'","tedy","tema","ten","tento","teto","ti","tim","timto","tipy","tisã­c","tisã­ce","tisíc","tisíce","to","tobä›","tobě","tohle","toho","tohoto","tom","tomto","tomu","tomuto","toto","troå¡ku","trošku","tu","tuto","tvoje","tvá","tvã¡","tvã©","två¯j","tvé","tvůj","ty","tyto","tä›","tå™eba","tå™i","tå™inã¡ct","téma","této","tím","tímto","tě","těm","těma","těmu","třeba","tři","třináct","u","uräitä›","určitě","uz","uå¾","už","v","vam","vas","vase","vaå¡e","vaå¡i","vaše","vaši","ve","vedle","veäer","večer","vice","vlastnä›","vlastně","vsak","vy","vám","vámi","vás","váš","vã¡m","vã¡mi","vã¡s","vã¡å¡","vå¡echno","vå¡ichni","vå¯bec","vå¾dy","více","však","všechen","všechno","všichni","vůbec","vždy","z","za","zatã­mco","zatímco","zaä","zač","zda","zde","ze","zpet","zpravy","zprávy","zpět","äau","ätrnã¡ct","ätyå™i","å¡est","å¡estnã¡ct","å¾e","čau","či","článek","článku","články","čtrnáct","čtyři","šest","šestnáct","že"]




    # Load the movie review data
    print('Loading Data')
    texts = text_helpers.load_slantour_data(data_folder_name, dataName)

    # Normalize text
    print('Normalizing Text Data')
    texts = text_helpers.normalize_text(texts, stops)
    print(len(texts))
    # Texts must contain at least 3 words
    #target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > window_size]
    #texts = [x for x in texts if len(x.split()) > window_size]
    #assert(len(target)==len(texts))

    # Build our data set and dictionaries
    print('Creating Dictionary')
    word_dictionary = text_helpers.build_dictionary(texts, vocabulary_size)
    word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
    text_data = text_helpers.text_to_numbers(texts, word_dictionary)

    # Get validation word keys
    valid_words = [word_dictionary_rev[1],word_dictionary_rev[10],word_dictionary_rev[100],word_dictionary_rev[1000]]
    valid_examples = [word_dictionary[x] for x in valid_words]

    print('Creating Model')
    # Define Embeddings:
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    doc_embeddings = tf.Variable(tf.random_uniform([len(texts), doc_embedding_size], -1.0, 1.0))

    # NCE loss parameters
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, concatenated_size],
                                                   stddev=1.0 / np.sqrt(concatenated_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Create data/target placeholders
    x_inputs = tf.placeholder(tf.int32, shape=[None, window_size + 1]) # plus 1 for doc index
    y_target = tf.placeholder(tf.int32, shape=[None, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Lookup the word embedding
    # Add together element embeddings in window:
    embed = tf.zeros([batch_size, embedding_size])
    for element in range(window_size):
        embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

    doc_indices = tf.slice(x_inputs, [0,window_size],[batch_size,1])
    doc_embed = tf.nn.embedding_lookup(doc_embeddings,doc_indices)

    # concatenate embeddings
    final_embed = tf.concat([embed, tf.squeeze(doc_embed, [1])],1)

    # Get loss from prediction
    #loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, final_embed, y_target, num_sampled, vocabulary_size))
    loss = tf.reduce_mean(tf.nn.nce_loss(
            weights = nce_weights,
            biases = nce_biases,
            inputs = final_embed,
            labels = y_target,
            num_sampled = num_sampled,
            num_classes = vocabulary_size))

    # Create optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate)
    train_step = optimizer.minimize(loss)

    # Cosine similarity between words
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Create model saving operation
    saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings})

    #Add variable initializer.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Run the skip gram model.
    print('Starting Training')
    loss_vec = []
    loss_x_vec = []
    for i in range(generations):
        batch_inputs, batch_labels = text_helpers.generate_batch_data(text_data, batch_size,
                                                                      window_size, method='doc2vec')
        feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}

        # Run the train step
        sess.run(train_step, feed_dict=feed_dict)

        # Return the loss
        if (i+1) % print_loss_every == 0:
            loss_val = sess.run(loss, feed_dict=feed_dict)
            loss_vec.append(loss_val)
            loss_x_vec.append(i+1)
            print('Loss at step {} : {}'.format(i+1, loss_val))

        # Validation: Print some random words and top 5 related words
        if (i+1) % print_valid_every == 0:
            sim = sess.run(similarity, feed_dict=feed_dict)
            for j in range(len(valid_words)):
                valid_word = word_dictionary_rev[valid_examples[j]]
                top_k = 5 # number of nearest neighbors
                nearest = (-sim[j, :]).argsort()[1:top_k+1]
                log_str = "Nearest to {}:".format(valid_word)
                for k in range(top_k):
                    close_word = word_dictionary_rev[nearest[k]]
                    log_str = '{} {},'.format(log_str, close_word)
                print(log_str)

        # Save dictionary + embeddings
        if (i+1) % save_embeddings_every == 0:
            # Save vocabulary dictionary
            with open(os.path.join(data_folder_name,'movie_vocab.pkl'), 'wb') as f:
                pickle.dump(word_dictionary, f)

            # Save embeddings
            model_checkpoint_path = os.path.join(os.getcwd(),data_folder_name,'doc2vec_movie_embeddings.ckpt')
            save_path = saver.save(sess, model_checkpoint_path)
            print('Model saved in file: {}'.format(save_path))


    final_embeddings = sess.run(doc_embeddings)
    embeddingsFname = "embeds/embed_doc2vec_"+str(window_size)+"_"+str(embedding_size)+".csv"
    np.savetxt(embeddingsFname, final_embeddings, fmt="%.6e")
    return final_embeddings

