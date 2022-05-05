from NeuralAgents.MetaModel import MetaModel
from NeuralAgents.transformer_keras.transformer_keras_AVR import Transformer_model
from NeuralAgents.PIXORModel import Transformer_lidar
from AVR.DataParser import DataParser, MEM_LENGTH, GAP_FRAMES, ACTOR_NUM, PEER_DATA_SIZE, N_ACTION, META_DATA_SIZE_IN_MODEL
from AVR.PCProcess import LidarPreprocessor

import os
import numpy as np
import argparse
import h5py

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

META = "META"
PIXOR = "PIXOR"
TRANSFORMER = "TRANS"
print("meta_data_size_in_model",META_DATA_SIZE_IN_MODEL)
print("n_action", N_ACTION)
class AgentTrainer(object):
    def __init__(self, train_dir, val_dir,  model_name=PIXOR, mem_length=1, display=None, batch_size=64, epoch=100):
        self._train_data_parser = DataParser(train_dir, display)
        self._val_data_parser = DataParser(val_dir, display)
        self._batch_size = batch_size
        self._train_data_dir = train_dir
        self._val_data_dir = val_dir
        self._agent = None
        self.lidar_preprocessor = LidarPreprocessor()
        self._display = display
        self._model_name = model_name
        self._epoch = epoch
        self.init_agent(mem_length)

    def init_agent(self, mem_length):
        if self._agent is not None:
            return
        if self._model_name==PIXOR:
            input_dim = self.lidar_preprocessor.lidar_depth_dim
            input_dim[2] = mem_length
            self._agent = Transformer_lidar(input_dim, n_action=N_ACTION, meta_size=META_DATA_SIZE_IN_MODEL, batch_size=self._batch_size, epoch=self._epoch)
            print("Initiating PIXOR agent, input dim", input_dim )

        # META data only models
        if self._model_name == META:
            input_dim = [28, 5, 7]
            self._agent = MetaModel(input_dim, n_action=3, batch_size=self._batch_size, epoch=self._epoch)
        if self._model_name == TRANSFORMER:
            # input_dim = [ACTOR_NUM*mem_length, PEER_DATA_SIZE]
            input_dim = [ACTOR_NUM, PEER_DATA_SIZE * mem_length]
            print("Initiating Transformer agent, input dim", input_dim )
            self._agent = Transformer_model(input_dim, n_action=N_ACTION, meta_size=META_DATA_SIZE_IN_MODEL, batch_size=self._batch_size, epoch=self._epoch)

    def prep_meta_data(self, lidar_fp, mem_length=1, gap_frames=2):

        # data_format = "ego_lidar_depth"
        data_format = "merged_lidar_depth"
        concat_axis=3
        if self._model_name != PIXOR:
            data_format = "peer_data"
            concat_axis=2
        start = gap_frames * mem_length + 1  # was gap_frames * mem_length
        end = -1 - gap_frames * mem_length  # was -1

        hf = h5py.File(lidar_fp, 'r')
        peer_data = None
        for i in range(mem_length):
            s = start - gap_frames * i
            ee = end - gap_frames * i
            if peer_data is None:
                peer_data = hf.get(data_format)[s:ee]
            else:
                peer_data = np.concatenate([peer_data, hf.get(data_format)[s:ee]], axis=concat_axis)

        ego_meta = hf.get('ego_meta')[start:end]
        X = [peer_data, ego_meta[:, :, 3:]]
        # X = [peer_data, hf.get('peer_data_mask')[start:end], ego_meta[:, :, 3:]]

        # Y = hf.get('ego_actions')[start:end]

        current_geom = ego_meta[:, :, 0:3]
        output_geom = None
        for i in range(mem_length):
            delta = gap_frames * (i+1)
            future_geom = hf.get('ego_meta')[start+delta:end+delta][:, :, 0:3]
            future_geom = future_geom - current_geom
            future_geom = np.reshape(future_geom, (future_geom.shape[0], future_geom.shape[2]))
            if output_geom is None:
                output_geom = future_geom
            else:
                output_geom = np.concatenate([output_geom, future_geom], axis=1)



        # # auxillary losses
        # y_aux = hf.get('ego_meta')[start+gap_frames:end+gap_frames][:, :, 3:5]
        # print(y_aux.shape)
        # y_aux = np.reshape(y_aux, (y_aux.shape[0], y_aux.shape[2]))
        # print(y_aux.shape)
        # y = np.concatenate([Y, y_aux], axis=1)
        # print(y.shape)
        y = output_geom
        return X, y

    def train_model(self, mem_length=1, gap_frames=2):

        X = None
        Y = None
        for e in self._train_data_parser._episodes:
            data_file = e + ".h5"
            data_fp = os.path.join(self._train_data_dir, e, data_file)
            if not os.path.exists(data_fp):
                # self._train_data_parser.parse_episode(e, self._model_name)
                self._train_data_parser.parse_episode(e, self._model_name)
            episode_X, episode_Y = self.prep_meta_data(data_fp, mem_length=mem_length, gap_frames=gap_frames)
            if X is None:
                X = episode_X
                Y = episode_Y
            else:
                for i in range(len(X)):
                    X[i] = np.concatenate([X[i], episode_X[i]])
                Y = np.concatenate([Y, episode_Y])

        X_val = None
        Y_val = None
        for e in self._val_data_parser._episodes:
            lidar_file = e + ".h5"
            lidar_fp = os.path.join(self._val_data_dir, e, lidar_file)
            if not os.path.exists(lidar_fp):
                # self._val_data_parser.parse_episode(e)
                self._val_data_parser.parse_episode(e)

            episode_X, episode_Y = self.prep_meta_data(lidar_fp, mem_length=mem_length, gap_frames=gap_frames)
            if X_val is None:
                X_val = episode_X
                Y_val = episode_Y
            else:
                for i in range(len(X_val)):
                    X_val[i] = np.concatenate([X_val[i], episode_X[i]])
                Y_val = np.concatenate([Y_val, episode_Y])

        print("Training on {} samples".format(Y.shape[0]))
        print("x[0][0] shape: ",X[0][0].shape, "Len x[0]", len(X[0]))
        #print(X[0][0])
        print("x[1][0] shape: ",X[1][0].shape, "Len x[1]", len(X[1]))
        print("y[0] shape", Y[0].shape, "Len y", len(Y))
        print("y[0]",Y[0])
        '''
        history = self._agent.train(X, Y, X_val, Y_val)
        # debug
        f = open("./fit_loss_history.txt", "a")
        f.write("TrainingLoss: {}\n".format(str(history.history["loss"])))
        f.write("ValLoss: {}\n".format(str(history.history["val_loss"])))
        f.close()

        self._agent.save("./{}.ckpt_{}".format(self._agent.id, self._agent.steps + self._agent._epoch))
        self._agent.save("./{}.ckpt".format(self._agent.id))
        '''

    def eval_model(self, mem_length=1, gap_frames=2):
        """Trace by trace"""
        for e in self._val_data_parser._episodes:
            lidar_file = e + ".h5"
            lidar_fp = os.path.join(self._val_data_dir, e, lidar_file)
            if not os.path.exists(lidar_fp):
                # self._val_data_parser.parse_episode(e)
                self._val_data_parser.parse_episode(e)

            X, Y = self.prep_meta_data(lidar_fp, mem_length, gap_frames=gap_frames)

            val_loss = self._agent.eval(X, Y)
            f = open("./val_loss.txt", "a")
            f.write("{}: {}\n".format(e, val_loss))
            f.close()

            # debug
            f = open("./val_loss.txt", "w")
            f.write("{}, {}, {}\n".format("agent_output", "oracal_output", "difference"))
            for i in range(X[0].shape[0]):
                frame_input = []
                for j in range(len(X)):
                    frame_input.append(X[j][i:i+1])

                # print(frame_input[0].shape)
                # print(frame_input[1].shape)
                agent_control = self._agent.predict(frame_input)
                oracal_control = Y[i]
                f.write("{},\n {},\n {}\n".format(agent_control, oracal_control, agent_control-oracal_control))
            f.close()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--train', default=None, help='training data dir')
    PARSER.add_argument('--val', default=None, help='validation data dir')
    PARSER.add_argument('--model', default=PIXOR, help='which model to train, (PIXOR, META)')
    PARSER.add_argument('--epoch', default=300, type=int, help='Number of training epoch')
    ARGUMENTS = PARSER.parse_args()

    trainer = AgentTrainer(ARGUMENTS.train, ARGUMENTS.val, ARGUMENTS.model, mem_length=MEM_LENGTH, epoch=ARGUMENTS.epoch)

    if ARGUMENTS.train is not None:
        trainer.train_model(mem_length=MEM_LENGTH, gap_frames=GAP_FRAMES)
    if ARGUMENTS.val is not None:
        trainer.eval_model(mem_length=MEM_LENGTH, gap_frames=GAP_FRAMES)



