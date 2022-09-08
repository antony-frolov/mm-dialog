from ipywidgets import Output, Button, Layout, HBox, VBox
from IPython.display import Image, display, clear_output, Markdown, HTML
from itertools import zip_longest


class LabelingTool:
    def __init__(
        self, dialog_dataset, indices, save2path=None,
        use_images=True, width=300, height=300,
        ignore_all_labeled=False
    ):
        self.dialog_dataset = dialog_dataset
        self.use_images = use_images
        self.ignore_all_labeled = ignore_all_labeled

        if self.use_images:
            self.image_dataset = self.dialog_dataset.image_dataset
            self.width = width
            self.height = height

        self.indices = indices
        self.save2path = save2path

        self.position = 0

        forward_button = Button(description="Next ➡️")
        forward_button.on_click(self._next_image)

        backward_button = Button(description="⬅️ Prev")
        backward_button.on_click(self._prev_image)

        labels = ['Yes ✅', 'No ❌']
        labels_buttons = [Button(description=label) for label in labels]
        for button in labels_buttons:
            button.on_click(self._save_label)

        self.image_frame = Output()
        self.utter_frame = Output(layout=Layout(max_width="400px"))
        self.context_frame = Output(layout=Layout(max_width="400px"))
        self.label_frame = Output()

        stop_button = Button(description='Stop ⛔')
        stop_button.on_click(self._stop)
        remove_button = Button(description='Remove label 🗑️')
        remove_button.on_click(self._remove_label)

        self.navigation_box = HBox([backward_button, forward_button])
        self.labels_box = HBox(labels_buttons)
        self.action_box = HBox([stop_button, remove_button])

        self.widgets = HBox([
            VBox([
                self.context_frame, self.utter_frame, self.label_frame,
                self.navigation_box, self.labels_box,
                self.action_box
            ], layout=Layout(align_items='center')),
            self.image_frame,
        ], layout=Layout(align_items='center', border='100px'))

        self.label2value = {'Yes ✅': True, 'No ❌': False}

    def _remove_label(self, button):
        idx = self.indices[self.position]
        self.dialog_dataset.image_like_flags[idx] = None
        with self.label_frame:
            clear_output(wait=True)
            display(Markdown(str(None)))

    def close_and_save(self, msg):
        self.widgets.close()
        clear_output(wait=True)
        print(msg)
        if self.save2path is not None:
            self.dialog_dataset.to_json(self.save2path)

    def _display_sample(self):
        if (not self.ignore_all_labeled and
                all(self.dialog_dataset[idx]['image_like'] is not None
                    for idx in self.indices)):
            self.close_and_save("All samples are labeled!")
            return

        if self.position < 0:
            self.position += 1
        if self.position >= len(self.indices):
            self.position -= 1
        idx = self.indices[self.position]
        item = self.dialog_dataset[idx]

        if self.use_images:
            path2image = self.image_dataset[item['image_idx']]['path2image']
            with self.image_frame:
                clear_output(wait=True)
                display(Image(path2image, height=200))

        with self.utter_frame:
            clear_output(wait=True)
            display(HTML(item['utter']))

        with self.context_frame:
            clear_output(wait=True)
            for line in (
                    ['A: ', 'B: '][i] + line
                    for pair in zip_longest(item['context'][::2], item['context'][1::2])
                    for i, line in enumerate(pair) if line is not None
            ):
                display(Markdown(line))

        with self.label_frame:
            clear_output(wait=True)
            display(Markdown(str(item['image_like'])))

    def _next_image(self, button=None):
        self.position += 1
        self._display_sample()

    def _prev_image(self, button=None):
        self.position -= 1
        self._display_sample()

    def _save_label(self, button):
        idx = self.indices[self.position]
        self.dialog_dataset.image_like_flags[idx] = self.label2value.get(button.description)
        self._next_image()

    def _stop(self, button):
        if all(self.dialog_dataset[idx]['image_like'] is not None
               for idx in self.indices):
            msg = "All samples are labeled!"
        else:
            msg = "Labeling is stopped!"
        self.close_and_save(msg)

    def label_samples(self):
        display(self.widgets)
        self._display_sample()
