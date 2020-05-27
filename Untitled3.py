
# coding: utf-8

# In[62]:


from fastai.vision import *


# In[63]:


classes = ['With-Mask','WithOut-Mask']


# In[66]:


path=Path('storage/data/mask')


# In[67]:


path.ls()


# In[68]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# In[69]:


data.classes


# In[70]:


data.show_batch(rows=3, figsize=(7,8))


# In[71]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[72]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[73]:


learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))


# In[75]:


learn.lr_find()


# In[76]:


learn.recorder.plot()


# In[77]:


learn.fit_one_cycle(6, max_lr=slice(3e-2,3e-1))


# In[79]:


learn.lr_find()


# In[80]:


learn.recorder.plot(),
learn.recorder.plot_lr(show_moms=True)


# In[81]:


learn.recorder.plot_losses()


# In[82]:


learn.fit_one_cycle(4, max_lr=slice(3e-3,3e-2))


# In[83]:


interp = ClassificationInterpretation.from_learner(learn)


# In[84]:


interp.plot_confusion_matrix()


# In[85]:


learn.export()


# In[98]:


defaults.device = torch.device('cpu')


# In[106]:


img = open_image(path/'3.jpg')
img


# In[107]:


learn = load_learner(path)


# In[108]:


pred_class,pred_idx,outputs = learn.predict(img)
pred_class ,

