
if __name__ == '__main__':
    import sys
    from model import load_frozen_model
    from main import play

    model_names = sys.argv[1:]
    if len(model_names) != 2:
        print('Please pass two model names.')
        sys.exit()
    
    models = [load_frozen_model(name) for name in model_names]
    
    play(models[0], models[1], output=True)
    
    #print(' ================ TEST ================== ')
    #b, m, r = play_parallel_with_results(models[1], models[0], 2, num_games=2)
    #for i in range(len(b)):
    #    pretty_print_board(b[i])
    #    print('Move', m[i], ' Reward', r[i])
