# Matplotlib utility functions.

export add_arrow

"Add an arrow to the object `line` from Matplotlib."
function add_arrow(line, size=12)

   color = line.get_color()
   xdata, ydata = line.get_data()

   for i = 1:2
      start_ind = length(xdata) ÷ 3 * i
      end_ind = start_ind + 1
      line.axes.annotate("",
         xytext=(xdata[start_ind], ydata[start_ind]),
         xy=(xdata[end_ind], ydata[end_ind]),
         arrowprops=Dict("arrowstyle"=>"-|>", "color"=>color),
         size=size
      )
   end

end