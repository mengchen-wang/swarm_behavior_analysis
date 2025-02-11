import xlsxwriter
import os

def generate(num):
    num_freq = 3 
    num_field = 5
    num_phase_z = 4
    num_phase_y = 4
    total = num_freq * num_freq * num_freq * num_field * num_field * num_field * num_phase_z * num_phase_y 
    copy=num

    total = total / num_freq
    fx = num // total
    num = num % total
    fx=3+2*fx

    total = total / num_freq
    fy = num // total
    num = num % total
    fy=3+2*fy

    total = total / num_freq
    fz = num // total
    num = num % total
    fz=3+2*fz

    total = total / num_field
    Bx = num // total
    num = num % total
    Bx=2*Bx

    total = total / num_field
    By = num // total
    num = num % total
    By=2*By

    total = total / num_field
    Bz = num // total
    num = num % total
    Bz=2*Bz

    total = total / num_phase_y
    phase_y = num // total
    num = num % total
    phase_y=phase_y*30

    total = total / num_phase_z
    phase_z = num // total
    num = num % total
    phase_z=phase_z*90
    return fx, fy, fz, Bx, By, Bz, phase_y, phase_z

def construct_library():
    wb=xlsxwriter.Workbook('图像.xlsx')
    ws=wb.add_worksheet('图像结果')
    ws.set_column('A:A',25)
    ws.set_column(0, 5, 15)
    headings=['20图像','22图像','24图像','26图像','60图像','100图像','Bx','By','Bz','fx','fy','fz','phase_y', 'phase_z']
    ws.set_tab_color('red')
    head_format=wb.add_format({'bold':1,'fg_color':'cyan','align':'center','font_name':u'微软雅黑','valign':'vcenter'})
    cell_format=wb.add_format({'bold':0,'align':'center','font_name':u'微软雅黑','valign':'vcenter'})
    ws.write_row('A1',headings,head_format)
    k = 0
    save_dir = r'./10uL/Frames' 
    for image in os.listdir(save_dir):
        if image!=".DS_S" and image!=".DS_Store":
            imagedir = os.path.join(save_dir, image)
            dir = os.path.join(imagedir, "20.jpg")
            dir2 = os.path.join(imagedir, "22.jpg")
            dir3 = os.path.join(imagedir, "24.jpg")
            dir4 = os.path.join(imagedir, "26.jpg")
            dir5 = os.path.join(imagedir, "60.jpg")
            dir6 = os.path.join(imagedir, "100.jpg")
            ws.set_row(k+1,60)
            ws.insert_image('A'+str(k+2), dir, {'x_scale':0.1,'y_scale':0.1})
            ws.insert_image('B'+str(k+2), dir2, {'x_scale':0.1,'y_scale':0.1})
            ws.insert_image('C'+str(k+2), dir3, {'x_scale':0.1,'y_scale':0.1})
            ws.insert_image('D'+str(k+2), dir4, {'x_scale':0.1,'y_scale':0.1})
            ws.insert_image('E'+str(k+2), dir5, {'x_scale':0.1,'y_scale':0.1})
            ws.insert_image('F'+str(k+2), dir6, {'x_scale':0.1,'y_scale':0.1})
            fx, fy, fz, Bx, By, Bz, phase_y, phase_z = generate(int(image))
            ws.write(k+1,6,Bx,cell_format)
            ws.write(k+1,7,By,cell_format)
            ws.write(k+1,8,Bz,cell_format)
            ws.write(k+1,9,fx,cell_format)
            ws.write(k+1,10,fy,cell_format)
            ws.write(k+1,11,fz,cell_format)
            ws.write(k+1,12,phase_y,cell_format)
            ws.write(k+1,13,phase_z,cell_format)
            k+=1
    wb.close()

if __name__ == "__main__":
    construct_library()