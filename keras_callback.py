import sys
import time
import keras
import keras_io
 
class CustomCallback(keras.callbacks.Callback):

    def __init__(self, total_epochs, epoch_multiple = 1, autosave = True):
        self.losses = []
        self.epoch_multiple = epoch_multiple
        self.total_epochs = total_epochs
        self.autosave = autosave

    def on_train_begin(self, logs={}):
        print '\n### Training Model ###'
        self.start_t = time.time()
        return
 
    def on_train_end(self, logs={}):
        print '\n### Finished Training ###'
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        if (epoch == 0):
            self.losses.append(logs.get('loss'))
            return #Don't care about 0 progress and don't want to divide by 0 in ETA calcs

        if (epoch % self.epoch_multiple == 0):
            #Clear any line
            sys.stdout.write('                                                                                 \r')
            sys.stdout.flush()

            pctDone = (float(epoch) / float(self.total_epochs)) * 100
            eta = ((time.time() - self.start_t) / float(epoch)) * (self.total_epochs - epoch)
            eta_mins = int(eta)/60
            eta_hrs  = eta_mins/60
            eat_sec  = int(eta) % 60
            eta_mins = eta_mins-(eta_hrs*60)
            sys.stdout.write('Epoch ' +
                str(epoch) + '/' +
                str(self.total_epochs) + ' (' + format(pctDone, '0.1f') + '%) | ETA: ' + str(eta_hrs) + 'h:' + str(eta_mins) + 'm:' + str(eat_sec) + 's | Loss:' +
                str(logs.get('loss')) + '\r')
            sys.stdout.flush()

        #Save model if log loss is less than prior
        if (self.autosave) and (logs.get('loss') < self.losses[-1]):
            print ''
            keras_io.save_model(self.model, 'model_autosave')

        self.losses.append(logs.get('loss'))
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return