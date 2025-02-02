import time
from options.train_options import TrainOptions
import data as Dataset
from model import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()
    # create a dataset
    dataset = Dataset.create_dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('training images = %d' % dataset_size)

    # for i, data in enumerate(dataset):
    #     print(i)

    # create a model
    model = create_model(opt)
    # model = model.to()  
    # create a visualizer
    visualizer = Visualizer(opt)
    # training flag
    keep_training = True
    max_iteration = opt.niter+opt.niter_decay
    epoch = 0
    total_iteration = opt.iter_count
    # total_iteration = -1
    # training process
    while(keep_training):
        epoch_start_time = time.time()
        # epoch+=1
        # print('\n Training epoch: %d' % epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iteration += 1
            model.set_input(data)
            model.optimize_parameters()
            # visualizer.display_current_results(model.get_current_visuals(), epoch)
            # visualizer.display_current_results(model.get_current_visuals(), epoch)
            
            # print training loss and save logging information to the disk
            if total_iteration % opt.print_freq == 0:
                losses = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, total_iteration, losses, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(total_iteration, losses)
                    
            # display images on visdom and save images
            if total_iteration % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch)
                if hasattr(model, 'distribution'):
                    visualizer.plot_current_distribution(model.get_current_dis())


            if total_iteration % opt.eval_iters_freq == 0:
                model.eval() 
                if hasattr(model, 'eval_metric_name'):
                    eval_results = model.get_current_eval_results()  
                    visualizer.print_current_eval(epoch, total_iteration, eval_results)
                    if opt.display_id > 0:
                        visualizer.plot_current_score(total_iteration, eval_results)
                    
            # save the latest model every <save_latest_freq> iterations to the disk
            if total_iteration % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_iteration))
                model.save_networks('latest')

            # save the model every <save_iter_freq> iterations to the disk
            if total_iteration % opt.save_iters_freq == 0:
                print('saving the model of iterations %d' % total_iteration)
                model.save_networks(total_iteration)
                epoch+=1
                print('\n Training epoch: %d' % epoch)

            if total_iteration > max_iteration:
                keep_training = False
                break
            
            if total_iteration % opt.print_freq == 0:
                lr = model.optimizers[0].param_groups[0]['lr']
                print('learning rate=%.7f' % lr)

            model.update_learning_rate()

        print('\nEnd training')
